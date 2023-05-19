import clip
import torch
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import time
from util.transforms import build_se3_transform, se3_to_components
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import tf
import cv2

rospy.init_node('Test')


def main():
    map_dir = '/home/padfoot7/Desktop/RRC/CLIP_Project/lidar_lseg/data/lego_loam_map2/dump/'
    br = tf.TransformBroadcaster()
    

    # Clip IMG
    imgs_dir = '/home/padfoot7/Desktop/RRC/CLIP_Project/lidar_lseg/data/lego_loam_map2/lego_loam_images_png/'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px")
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)
    image = preprocess(Image.open("data/topomap/1.jpeg")).unsqueeze(0).to(device)
    feats_query_img = model.encode_image(image).cpu().to(torch.float32)
    
    # Clip Text
    text = clip.tokenize(["benches on the left of the road"]).to(device)
    feats_text = model.encode_text(text).cpu().to(torch.float32)

    idx_start, idx_end = 1455, 1598
    x_correction = 0
    x_clip = -80

    for i in range(10):
        br.sendTransform((-208, 144, 0),
                        tf.transformations.quaternion_from_euler(0, 0, 0),
                        rospy.Time.now(),
                        'base_link',
                        "map", )
        time.sleep(0.1)
    input('Continue?')
    
    correction = True
    with torch.no_grad():
        for i in range(idx_start, idx_end):
            tf_file = map_dir + str(i).zfill(6) +'/data'
            with open(tf_file, 'r') as f:
                data = f.readlines()
            tf_bl = [[float(x) for x in row.strip().split()] for row in data[2:6]]
            tf_bl = np.array(tf_bl)
            tf_bl_2_camera = build_se3_transform([0,0,0,0,0,np.pi/2])
            tf_bl = tf_bl_2_camera @ tf_bl
            x,y,z,r,p,yaw = se3_to_components(tf_bl)


            # Adding noise in the map
            x = x*1.5 + 104
            
            
            # Correction Step
            img = Image.open(f"{imgs_dir}{str(i).zfill(6)}.png")
            refimgage = preprocess(img).unsqueeze(0).to(device)
            feats_ref_img = model.encode_image(refimgage).cpu().to(torch.float32)
            sim = cosine_similarity(feats_ref_img, feats_text)[0]
            print(sim)
            # if sim>0.84:
            if sim>0.265:
                x_correction = x - x_clip

            if correction:
                x = x - x_correction

            
            
            print(x)

            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL) 
            cv2.imshow('frame', img)
            cv2.waitKey(20)

            br.sendTransform((x, y, z),
                    tf.transformations.quaternion_from_euler(r, p, yaw),
                    rospy.Time.now(),
                    'base_link',
                    "map")
            time.sleep(0.15)
                # pose = PoseStamped()

                # pose.header = Header()
                # pose.pose.position.x = x
                # pose.pose.position.y = y
                # pose.pose.position.z = z
                # pose.pose.orientation.x = r
                # pose.pose.orientation.y = p
                # pose.pose.orientation.z = yaw



if __name__=='__main__':
    main()