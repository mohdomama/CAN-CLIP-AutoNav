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
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-L/14@336px")
map_dir = '/home/padfoot7/Desktop/RRC/CLIP_Project/lidar_lseg/data/lego_loam_map2/dump/'
imgs_dir = '/home/padfoot7/Desktop/RRC/CLIP_Project/lidar_lseg/data/lego_loam_map2/lego_loam_images_png/'
cosine_similarity = torch.nn.CosineSimilarity(dim=1)


def select_k(feats, k, idxs):

    while len(idxs) > k:
        feats_current = feats[idxs]
        IG = []
        for i in range(len(idxs)):
            feats_i = feats_current[i]
            p_i = cosine_similarity(feats_i.reshape(1,-1), feats_current).sum()
            IG.append(1/p_i)
        IG = np.array(IG)
        least_ig = IG.argmin()
        idxs.pop(least_ig)

    return np.array(idxs)

def select_k_nextsim(feats, k, idxs):

    while len(idxs) > k:
        feats_current = feats[idxs]
        IG = []
        for i in range(len(idxs)):
            feats_i = feats_current[i]
            p_i = cosine_similarity(feats_i.reshape(1,-1), feats_current)
            p_i = p_i.sort()[0]
            IG.append(1/p_i[-2]) 
            # IG.append(1/p_i)
        IG = np.array(IG)
        least_ig = IG.argmin()
        idxs.pop(least_ig)

    return np.array(idxs)

def select_1(feats):
    IG = []
    for i in range(len(feats)):
        feats_i = feats[i]
        p_i = cosine_similarity(feats_i.reshape(1,-1), feats).sum()
        IG.append(1/p_i)

    IG = np.array(IG)
    least_ig = IG.argmin()
    return least_ig

def get_top_K_info_locally(idx_start, idx_end, k):
    all_feats = torch.load('data/all_feats.pt')
    feats_seq = all_feats[idx_start:idx_end]

    regions = np.linspace(0, len(feats_seq), k+1).astype(int)
    
    idxs = []
    for i in range(k):
        start, end = regions[i], regions[i+1]
        idx = select_1(feats_seq[start:end]) + start + idx_start
        idxs.append(idx)

    return np.array(idxs)


def get_top_K_info_idxs(idx_start, idx_end, k):
    all_feats = torch.load('data/all_feats.pt')
    feats_seq = all_feats[idx_start:idx_end]

    idxs = select_k(feats_seq, k, list(range(0, idx_end-idx_start)))
    idxs = idxs+idx_start

    return idxs

def get_top_K_info_nextsim(idx_start, idx_end, k):
    all_feats = torch.load('data/all_feats.pt')
    feats_seq = all_feats[idx_start:idx_end]

    idxs = select_k_nextsim(feats_seq, k, list(range(0, idx_end-idx_start)))
    idxs = idxs+idx_start

    return idxs

def get_total_information(lm_idxs):
    all_feats = torch.load('data/all_feats.pt')
    feats_lm = all_feats[lm_idxs]

    similarity = 0
    for feat in feats_lm:
        similarity += cosine_similarity(feat.reshape(1,-1), feats_lm).sum()
    return 1 / similarity, similarity

def get_total_information_nextsim(lm_idxs):
    all_feats = torch.load('data/all_feats.pt')
    feats_lm = all_feats[lm_idxs]

    similarity = 0
    for feat in feats_lm:
        x = cosine_similarity(feat.reshape(1,-1), feats_lm)
        x = x.sort()[0]
        similarity+=x[-2]
    return 1 / similarity, similarity


def main():

    idx_start, idx_end = 0, 190
    k = 20
    lm_ig = get_top_K_info_nextsim(idx_start, idx_end, k)
    ti_lm, sim = get_total_information_nextsim(lm_ig)
    print('Total Info IG, SIM: ', ti_lm, sim)

    lm_eq = np.linspace(idx_start, idx_end-1,k).astype(int)
    ti_eq , sim= get_total_information_nextsim(lm_eq)
    print('Total Info EQ, SIM: ', ti_eq, sim)

    lm_rd = np.random.choice(list(range(idx_start, idx_end)), k, replace=False)
    ti_rd, sim = get_total_information_nextsim(lm_rd)
    print('Total Info RD, SIM: ', ti_rd, sim)
    
    breakpoint()
    print('End')
    


def generate_all_feats():
    print('Hello')
    idx_start, idx_end = 0, 1885
    all_feats = []
    with torch.no_grad():
        for i in tqdm(range(idx_start, idx_end)):
            img = Image.open(f"{imgs_dir}{str(i).zfill(6)}.png")
            refimgage = preprocess(img).unsqueeze(0).to(device)
            feats_ref = model.encode_image(refimgage).detach().cpu().to(torch.float32)
            all_feats.append(feats_ref)

    all_feats = torch.vstack(all_feats)
    torch.save(all_feats, 'data/all_feats.pt')
    print('End')
            

if __name__=='__main__':
    main()