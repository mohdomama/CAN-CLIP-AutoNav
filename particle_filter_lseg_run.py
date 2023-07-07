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
import open3d as o3d
from dataclasses import dataclass
import tyro
from lseg import LSegNet
from util.o3d_util import visualize_multiple_pcd


rospy.init_node('Test')

@dataclass
class ProgramArgs:
    checkpoint_path: str =  "../CLIP_Project/lseg-minimal" + "/examples" +"/checkpoints" + "/lseg_minimal_e200.ckpt"
    backbone: str = "clip_vitl16_384"
    num_features: int = 256
    arch_option: int = 0
    block_depth: int = 0
    activation: str = "lrelu"
    sequence: str = "00"
    crop_size: int = 480

device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-L/14@336px")

map_dir = '/home/padfoot7/Desktop/RRC/CLIP_Project/lidar_lseg/data/lego_loam_map1/dump/'
map_path = '../CLIP_Project/lidar_lseg/data/lego_loam_map1/'
imgs_dir = '/home/padfoot7/Desktop/RRC/CLIP_Project/lidar_lseg/data/lego_loam_map1/lego_loam_images_png/'

args = tyro.cli(ProgramArgs)
net = LSegNet(
    backbone=args.backbone,
    features=args.num_features,
    crop_size=args.crop_size,
    arch_option=args.arch_option,
    block_depth=args.block_depth,
    activation=args.activation,
)

net.load_state_dict(torch.load(str(args.checkpoint_path)))
net.eval()
net.cuda()
clip_text_encoder = net.clip_pretrained.encode_text

cosine_similarity = torch.nn.CosineSimilarity(dim=2)

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


def get_img_feat(img, net):
    '''
    img -> RGB (0, 255)
    '''
    # Load the input image
    with torch.no_grad():
        print(f"Original image shape: {img.shape}")
        img = cv2.resize(img, (640, 480))
        img = torch.from_numpy(img).float() / 255.0
        img = img[..., :3]  # drop alpha channel, if present
        img = img.cuda()
        img = img.permute(2, 0, 1)  # C, H, W
        img = img.unsqueeze(0)  # 1, C, H, W
        print(f"Image shape: {img.shape}")

        # Extract per-pixel CLIP features (1, 512, H // 2, W // 2)
        img_feat = net.forward(img)
        # Normalize features (per-pixel unit vectors)
        img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)
        print(f"Extracted CLIP image feat: {img_feat_norm.shape}")
    return img_feat_norm

def read_lego_loam_data_file(map_path, keyframe_id):
    filename = map_path + 'dump/' + keyframe_id + '/data'
    with open(filename, 'r') as f:
        lines = f.readlines()

    odom = []
    for i in range(7, 11):
        row = [float(x) for x in lines[i].strip().split()]
        odom.append(row)

    return np.array(odom)

def velo_to_cam(pcd, P, T):
        out =  (P @ T @ pcd.T).T
        out[:, 0] = out[:, 0] / out[:, 2]
        out[:, 1] = out[:, 1] / out[:, 2]
        out[:, 2] = out[:, 2] / out[:, 2]
        return out


def get_projection_matrices():
    # From rostopic echo /zed/zed_node/left/camera_info
    K = [336.3924560546875, 0.0, 307.7497863769531, 0.0, 336.3924560546875, 165.84719848632812, 0.0, 0.0, 1.0]
    K = np.array(K).reshape((3,3))
    K = np.hstack([K, np.array([0,0,0]).reshape(3,1)])

    T_camo_c = build_se3_transform([0.13, -0.35, -0.7, 0, 0, 0])
    P = K @ T_camo_c

    Tr = [0, -1,  0, 0, 
        0,  0, -1, 0, 
        1,  0,  0, 0]
    Tr = np.array(Tr).reshape(3,4)
    Tr_vel_camo = np.vstack([Tr, [0,0,0,1]])
    return P, Tr_vel_camo


def calculate_closest(feats, poses_feat, particles):
    closest_idx = torch.cdist(particles[:, :2, -1], poses_feat[:, :2, -1]).argmin(axis=1)
    distances = torch.cdist(particles[:, :2, -1], poses_feat[:, :2, -1]).min(axis=1)[0]
    return feats[closest_idx], distances


def observation_model_closest_pcd(pcd_feat, pcd_feat_map, pcd_map, pcd, particles):
    sim = cosine_similarity(pcd_feat.unsqueeze(1), pcd_feat_map.unsqueeze(0))
    weights = []

    for particle in particles:
        pcd_map_pf = (particle@pcd_map.T).T # lm_can in particle frame
        
        dis = torch.cdist(pcd[:, :3], pcd_map_pf[:, :3])
        wt_dis = 1/(dis+0.01)
        
        wt_matrix = sim * wt_dis
        wt_matrix = wt_matrix.mean(dim=1)
        wt = wt_matrix.mean()

        # Find closest
        weights.append(wt)
    
    weights = torch.tensor(weights)
    weights = weights * 20  # Temperature
    weights = F.softmax(weights, dim=-1, )
    return weights



def upsample_feat_vec(feat, target_shape):
    return torch.nn.functional.interpolate(
        feat, target_shape, mode="bilinear", align_corners=True
    )



def main():
    pcd_feat_path = '/home/padfoot7/Desktop/RRC/CLIP_Project/lidar_lseg/data/lego_loam_map1/pcd_feat_map.npy'
    pcd_path = '/home/padfoot7/Desktop/RRC/CLIP_Project/lidar_lseg/data/lego_loam_map1/pcd_map.npy'
    pcd_feat_map = np.load(pcd_feat_path)
    pcd_map = np.load(pcd_path)

    ig_idxs = np.load('data/idxs_pcd_nextsim_tiny.npy')

    ## Use random idxs instead
    # ig_idxs = np.random.choice(range(0,6000), ig_idxs.shape[0], replace=False)

    pcd_feat_map = pcd_feat_map[ig_idxs]
    pcd_map = pcd_map[ig_idxs]


    br = tf.TransformBroadcaster()
    P, Tr_vel_camo = get_projection_matrices()

    T_static = build_se3_transform([0,0,0,0,0,1.57]) 

    idx_start, idx_end = 0, 91
    tf_gt_old = build_se3_transform([0,0,0,0,0,0])
    particleFilter = ParticleFilter(300)

    with torch.no_grad():
        for i in range(idx_start, idx_end):
            keyframe_id = str(i).zfill(6)
            pcd = o3d.io.read_point_cloud(map_path+'dump/' +  keyframe_id + '/' + 'cloud.pcd')
            pcd = np.asarray(pcd.points)
            img = np.load(map_path+'lego_loam_images/' + keyframe_id + '.npy')

            ##################
            # Begin Projection
            
            pcd =  np.hstack([pcd, np.ones((pcd.shape[0], 1))])
            mask = np.ones(pcd.shape[0])
            mask[np.where(pcd[:,0]<1)[0]] = 0
            
            pts_cam = velo_to_cam(pcd, P, Tr_vel_camo)
            pts_cam = np.array(pts_cam, dtype=np.int32)  # 0th column is img height (different from kitti)

            # #  Filter pts_cam to get only the point in image limits
            # # There should be a one liner to do this.
            mask[np.where(pts_cam[:,0] >=img.shape[1])[0]] = 0
            mask[np.where(pts_cam[:,0] <0)[0]] = 0
            mask[np.where(pts_cam[:,1] >=img.shape[0])[0]] = 0
            mask[np.where(pts_cam[:,1] <0)[0]] = 0
            mask_idx = np.where([mask>0])[1]  # Somehow this returns a tuple of len 2
            tic = time.time()
            img_feat = get_img_feat(img, net)
            toc = time.time()
            print('Net Time: ', toc-tic)
            img_feat = upsample_feat_vec(img_feat, img.shape[:-1])


            pcd_feat = torch.zeros((pcd.shape[0], 512), dtype=torch.float32).cuda()
            img_feat = img_feat.permute((0,2,3,1))           
            pcd_feat[mask_idx] =  img_feat[0][pts_cam[mask_idx, 1], pts_cam[mask_idx, 0], :]
            pcd_feat = pcd_feat[mask_idx]
            pcd = pcd[mask_idx]
            ##### Done Projection #######

            
            tf_file = map_dir + str(i).zfill(6) +'/data'
            with open(tf_file, 'r') as f:
                data = f.readlines()
            
            tf_gt = get_tf_bl(data)
            
            tf_motion = np.linalg.pinv(tf_gt_old) @ tf_gt

            motion_model = partial(motion_model_naive, tf_motion)
            particleFilter.motion_update(motion_model)

            
            observation_model = partial(observation_model_closest_pcd, pcd_feat.detach().cpu(), torch.tensor(pcd_feat_map), torch.tensor(pcd_map), torch.tensor(pcd))
            tic = time.time()
            particleFilter.observation_update(observation_model)
            toc = time.time()
            print('Obs Model Time: ', toc-tic)
            particleFilter.resample()
            
            tf_gt_old = tf_gt
            send_transform(br, tf_gt)
            publish_particles(particleFilter.particles)

            # # Image Display
            # img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            # cv2.namedWindow('frame', cv2.WINDOW_NORMAL) 
            # cv2.imshow('frame', img)
            # cv2.waitKey(20)

            



if __name__=='__main__':
    # get_lang_feature_poses(idx=2)
    main()