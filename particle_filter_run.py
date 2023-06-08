import clip
import torch
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import time
from util.transforms import build_se3_transform, se3_to_components
import rospy
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import Header
import tf
from tf.transformations import euler_from_quaternion,  quaternion_from_euler
import cv2
from particle_filter import ParticleFilter
from functools import partial
import torch.nn.functional as F
from information_gain import get_top_K_info_idxs, get_top_K_info_locally
from tqdm import tqdm
torch.manual_seed(0)


rospy.init_node('Test')
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px")

map_dir = '/home/padfoot7/Desktop/RRC/CLIP_Project/lidar_lseg/data/lego_loam_map2/dump/'
imgs_dir = '/home/padfoot7/Desktop/RRC/CLIP_Project/lidar_lseg/data/lego_loam_map2/lego_loam_images_png/'

cosine_similarity = torch.nn.CosineSimilarity(dim=1)

particles_publisher = rospy.Publisher("/particles", PoseArray, queue_size=1, latch=False)

def get_tf_bl(data):
    tf_bl = [[float(x) for x in row.strip().split()] for row in data[2:6]]
    tf_bl = np.array(tf_bl)
    tf_bl_2_camera = build_se3_transform([0,0,0,0,0,np.pi/2])
    tf_bl = tf_bl_2_camera @ tf_bl
    return tf_bl

def publish_particles(particles):
    pose_array_msg = PoseArray()

    pose_array_msg.header = Header()
    pose_array_msg.header.stamp = rospy.Time.now()
    pose_array_msg.header.frame_id = 'map'

    for particle in particles:
        x,y,z,r,p,yaw = se3_to_components(particle.numpy())
        msg = Pose()
        msg.position.x = x
        msg.position.y = y
        msg.position.z = z

        quat = quaternion_from_euler(0, 0, yaw)
        msg.orientation.x = quat[0]
        msg.orientation.y = quat[1]
        msg.orientation.z = quat[2]
        msg.orientation.w = quat[3]
        pose_array_msg.poses.append(msg)
    
    particles_publisher.publish(pose_array_msg)

def send_transform(br, tf_tosend):
    x,y,z,r,p,yaw = se3_to_components(tf_tosend)
    # Sending Transform
    br.sendTransform((x, y, z),
            tf.transformations.quaternion_from_euler(r, p, yaw),
            rospy.Time.now(),
            'base_link',
            "map")
    pass


def motion_model_naive(tf_motion, particles):
    particles = particles.numpy()@tf_motion
    xnoise = np.random.randn(particles.shape[0])
    ynoise = np.random.randn(particles.shape[0])
    particles[:, 0, -1] += xnoise
    particles[:, 1, -1] += ynoise
    
    return torch.tensor(particles)


def get_features_poses(idx_start, idx_end, k, method='ig'):

    if method == 'ig':
        # idxs = get_top_K_info_idxs(idx_start, idx_end, k=k)
        idxs = get_top_K_info_locally(idx_start, idx_end, k=k)

    if method == 'equal':
        idxs = np.linspace(idx_start, idx_end-1,k).astype(int)

    if method == 'random':
        idxs = np.random.choice(list(range(idx_start, idx_end)), k, replace=False)
    
    print('Usinge the folling landmarks:')
    print(idxs)
    feats = []
    poses = []
    with torch.no_grad():
        for idx in idxs:
            img = Image.open(f"{imgs_dir}{str(idx).zfill(6)}.png")
            img = preprocess(img).unsqueeze(0).to(device)
            img_feats = model.encode_image(img).cpu().to(torch.float32)
            feats.append(img_feats[0])

            tf_file = map_dir + str(idx).zfill(6) +'/data'
            with open(tf_file, 'r') as f:
                data = f.readlines()
            
            tf_feat = get_tf_bl(data)
            poses.append(torch.tensor(tf_feat))

    feats = torch.stack(feats)
    poses = torch.stack(poses)
    return feats, poses

def calculate_closest(feats, poses_feat, particles):
    closest_idx = torch.cdist(particles[:, :2, -1], poses_feat[:, :2, -1]).argmin(axis=1)
    distances = torch.cdist(particles[:, :2, -1], poses_feat[:, :2, -1]).min(axis=1)[0]
    return feats[closest_idx], distances


def observation_model_closest(feats_closest, feats_ref, distances, particles):
    weights = cosine_similarity(feats_closest, feats_ref)
    
    # test = torch.unique(weights.sort()[0])
    # if test[-1] == 1:
    #     print('Diff: ', test[-1]-test[-2])
    #     print('Last Five: ', test[-5:])
    # weights = weights / 0.3
    weights = weights / 6

    
    dw = 1/(distances+0.1)
    dw / 10
    # dw[distances>40] = 0
    dweights =  F.softmax(dw, dim=-1, )

    # weights = weights + dweights
    weights = weights

    weights = F.softmax(weights, dim=-1, )

    
    return weights


def main():
    br = tf.TransformBroadcaster()

    idx_start, idx_end = 0, 101
    drift_total = np.zeros((3,2,101))
    stattest_number = 1
    for stattest in range(stattest_number):
        print(stattest)
        drift_all = []
        for method in ['ig', 'equal', 'random']:
            feats, poses_feat = get_features_poses(idx_start, idx_end,k=20, method=method)
            tf_gt_old = build_se3_transform([0,0,0,0,0,0])
            particleFilter = ParticleFilter(600)

            pose_particles_mean, pose_particles_median = [], []
            pose_gt = []
            with torch.no_grad():
                for i in tqdm(range(idx_start, idx_end)):
                    tf_file = map_dir + str(i).zfill(6) +'/data'
                    with open(tf_file, 'r') as f:
                        data = f.readlines()
                    
                    tf_gt = get_tf_bl(data)
                    
                    # TODO: Add noise to tf_motion
                    tf_motion = np.linalg.pinv(tf_gt_old) @ tf_gt

                    motion_model = partial(motion_model_naive, tf_motion)
                    particleFilter.motion_update(motion_model)

                    feats_closest, distances = calculate_closest(feats, poses_feat, particleFilter.particles)
                    img = Image.open(f"{imgs_dir}{str(i).zfill(6)}.png")
                    refimgage = preprocess(img).unsqueeze(0).to(device)
                    feats_ref = model.encode_image(refimgage).cpu().to(torch.float32)
                    
                    observation_model = partial(observation_model_closest, feats_closest, feats_ref, distances)
                    particleFilter.observation_update(observation_model)
                    particleFilter.resample()
                    
                    tf_gt_old = tf_gt
                    send_transform(br, tf_gt)
                    publish_particles(particleFilter.particles)

                    # Compare with gt
                    particles_x = np.array([particle[0,-1] for particle in particleFilter.particles])
                    particles_y = np.array([particle[1,-1] for particle in particleFilter.particles])

                    pose_particles_mean.append([particles_x.mean(), particles_y.mean()])
                    pose_particles_median.append([np.median(particles_x), np.median(particles_y)])
                    pose_gt.append([tf_gt[0,-1], tf_gt[1,-1]])

            pose_particles_mean = np.array(pose_particles_mean)
            pose_particles_median = np.array(pose_particles_median)
            pose_gt = np.array(pose_gt)
            drift_mean = np.linalg.norm(pose_particles_mean-pose_gt, axis=1)
            drift_median = np.linalg.norm(pose_particles_median-pose_gt, axis=1)
            drift_all.append([drift_mean, drift_median])
            
        drift_all = np.array(drift_all)
        drift_total = drift_total+drift_all

    # Plotting mean
    drift_total = drift_total/stattest_number
    plt.plot(drift_total[0][0], 'g')
    plt.plot(drift_total[1][0], 'b')
    plt.plot(drift_total[2][0], 'r')
    plt.show()

    # Plotting mean
    plt.plot(drift_total[0][1], 'g')
    plt.plot(drift_total[1][1], 'b')
    plt.plot(drift_total[2][1], 'r')
    plt.show()


if __name__=='__main__':
    main()