import os
import glob
import cv2
import numpy as np
from pathlib import Path


def blockSampleFusion(estimate_block, o3d_block):
    assert estimate_block.size == o3d_block.size
    output_block = []
    o3d_block[o3d_block == np.inf] = np.nan
    
    estimate_depth_range = np.nanmax(estimate_block) - np.nanmin(estimate_block)
    o3d_depth_range = np.nanmax(o3d_block) - np.nanmin(o3d_block)
    e2o_ratio =  o3d_depth_range / estimate_depth_range
    print(estimate_block, "\n\n", o3d_block, "\n\n\n")
    estimate_block = estimate_block * e2o_ratio
    print(estimate_block, "\n\n", o3d_block, "\n\n\n")
    # for pix_col in range(o3d_block.size[0]):
    #     for pix_row in range(o3d_block.size[1]):

    
    return
    
    
def generateTrueDepth(data_dir, sizeof_sample_block = 16, valid_threshold = 0.3):
    estimate_depth_dir = data_dir + "relative_depth_predict/"
    o3d_depth_dir = data_dir + "unproject_depth/"
    
    edepth_path = sorted(glob.glob(estimate_depth_dir+"*.png"))
    odepth_path = sorted(glob.glob(o3d_depth_dir+ "*.npy"))
    
    file_idx = 0
    edepth_seq = Path(edepth_path[file_idx]).stem.split("_")[-1]
    odepth_seq = Path(odepth_path[file_idx]).stem.split("_")[-1]
    # print(edepth_seq, odepth_seq)
    assert edepth_seq == odepth_seq
    edepth = cv2.imread(edepth_path[file_idx], cv2.IMREAD_GRAYSCALE)
    edepth = cv2.bitwise_not(edepth)
    odepth = np.load(odepth_path[file_idx])
    # print(edepth,"\n", odepth)
    assert edepth.shape == odepth.shape
    
    block_counter = 0
    valid_block_counter = 0
    for pix_col in range(0, edepth.shape[0], sizeof_sample_block):
        for pix_row in range(0, edepth.shape[1], sizeof_sample_block):
            block_counter += 1
            esample_block = edepth[pix_col:min(edepth.shape[0], pix_col+sizeof_sample_block), \
                pix_row:min(edepth.shape[1], pix_row+sizeof_sample_block)]
            osample_block = odepth[pix_col:min(edepth.shape[0], pix_col+sizeof_sample_block), \
                pix_row:min(edepth.shape[1], pix_row+sizeof_sample_block)]
            # print(esample_block, osample_block)
            valid_ratio = 1 - np.count_nonzero(np.isinf(osample_block)) / sizeof_sample_block**2
            if valid_ratio > valid_threshold:
                valid_block_counter += 1
                merged_sample_block = blockSampleFusion(esample_block, osample_block)
                input()
    print("=> Total {} blocks, {} valid blocks, {:.3}% of Image Mended".format(
        block_counter, valid_block_counter, valid_block_counter/block_counter*100))
    

if __name__ == "__main__":
    data_dir = "inputs/sample/"
    generateTrueDepth(data_dir)