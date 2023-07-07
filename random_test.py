import torch
import numpy as np
from util.o3d_util import visualize_multiple_pcd


def main():
    pcd_feat_path = '/home/padfoot7/Desktop/RRC/CLIP_Project/lidar_lseg/data/lego_loam_map1/pcd_feat_map.npy'
    pcd_path = '/home/padfoot7/Desktop/RRC/CLIP_Project/lidar_lseg/data/lego_loam_map1/pcd_map.npy'
    pcd_feat_map = np.load(pcd_feat_path)
    pcd_map = np.load(pcd_path)

    ig_idxs = np.load('data/idxs_pcd_nextsim.npy')
    visualize_multiple_pcd([pcd_map[:60_000,:3], pcd_map[:,:3][ig_idxs] ]) 
    visualize_multiple_pcd([pcd_map[:,:3][ig_idxs] ]) 

    ig_idxs = np.load('data/idxs_pcd.npy')
    visualize_multiple_pcd([pcd_map[:,:3][ig_idxs] ]) 

    # Small

    ig_idxs = np.load('data/idxs_pcd_nextsim_small.npy')
    visualize_multiple_pcd([pcd_map[:,:3][ig_idxs] ]) 

    ig_idxs = np.load('data/idxs_pcd_small.npy')
    visualize_multiple_pcd([pcd_map[:,:3][ig_idxs] ]) 

    


    pass

def sparsemax(z):
    sum_all_z = sum(z)
    z_sorted = sorted(z, reverse=True)
    k = np.arange(len(z))
    k_array = 1 + k * z_sorted
    z_cumsum = np.cumsum(z_sorted) - z_sorted
    k_selected = k_array > z_cumsum
    k_max = np.where(k_selected)[0].max() + 1
    threshold = (z_cumsum[k_max-1] - 1) / k_max
    return np.maximum(z-threshold, 0)


if __name__=='__main__':
    main()