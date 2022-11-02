import os
import glob
import cv2
import numpy as np
from pathlib import Path
import copy


def blockSampleFusion(estimate_block, o3d_block):
    print(estimate_block, "\n", o3d_block)
    assert estimate_block.size == o3d_block.size
    print(o3d_block.size)
    output_block = o3d_block
    # get valid pixel in open3d block
    valid_pix = np.where(o3d_block != 0)
    # get the value of mapping valid pixel
    omapping_pix = o3d_block[valid_pix]
    emapping_pix = estimate_block[valid_pix]
    
    cv2.imshow('image',estimate_block)
    cv2.waitKey(0)
    cv2.imshow('image',o3d_block)
    cv2.waitKey(0)
    list_same = list(set(emapping_pix))
    # enum every estimate depth
    for source_depth in list_same:
        # caculate average of the same estimate depth 
        target_depth = np.average(omapping_pix[np.where(emapping_pix == source_depth)])
        print(target_depth)
        # mask = cv2.inRange(estimate_block, source_depth, source_depth)
        # o3d_block[mask > 0] = target_depth
        # set the inf value in open3d block to average according to estimate block
        output_block = np.array([[ target_depth if o3d_block[col, row] == 0 and estimate_block[col, row] == source_depth else output_block[col, row] \
                         for row in range(o3d_block.shape[1])] for col in range(o3d_block.shape[0])])
        print("output block:\n", output_block)
        
        cv2.imshow('image',output_block)
        cv2.waitKey(0)

        # print(output_block)
        # input()
    return
    
    
def generateTrueDepth(data_dir, sizeof_sample_block = 32, valid_threshold = 0.3):
    estimate_depth_dir = data_dir + "relative_depth_predict/"
    o3d_depth_dir = data_dir + "unproject_depth/"
    
    edepth_path = sorted(glob.glob(estimate_depth_dir+"*.png"))
    odepth_path = sorted(glob.glob(o3d_depth_dir+ "*.png"))
    
    file_idx = 0
    edepth_seq = Path(edepth_path[file_idx]).stem.split("_")[-1]
    odepth_seq = Path(odepth_path[file_idx]).stem.split("_")[-1]
    # print(edepth_seq, odepth_seq)
    assert edepth_seq == odepth_seq
    edepth = cv2.imread(edepth_path[file_idx], cv2.IMREAD_GRAYSCALE)
    edepth = cv2.bitwise_not(edepth)
    odepth = cv2.imread(odepth_path[file_idx], cv2.IMREAD_GRAYSCALE)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',edepth)
    cv2.waitKey(0)
    cv2.imshow('image',odepth)
    cv2.waitKey(0)
    # print(edepth,"\n", odepth)
    assert edepth.shape == odepth.shape
    out = np.zeros(edepth.shape)
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
            valid_ratio = 1 - np.count_nonzero(osample_block == 0) / sizeof_sample_block**2
            if valid_ratio > valid_threshold:
                valid_block_counter += 1
                merged_sample_block = blockSampleFusion(esample_block, osample_block)
                out[pix_col:min(edepth.shape[0], pix_col+sizeof_sample_block), \
                    pix_row:min(edepth.shape[1], pix_row+sizeof_sample_block)] = merged_sample_block
                cv2.imshow('image',out)
                cv2.waitKey(0)
            else:
                out[pix_col:min(edepth.shape[0], pix_col+sizeof_sample_block), \
                    pix_row:min(edepth.shape[1], pix_row+sizeof_sample_block)] = osample_block
            
    print("=> Total {} blocks, {} valid blocks, {:.3}% of Image Mended".format(
        block_counter, valid_block_counter, valid_block_counter/block_counter*100))
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imwrite('image.png',out)
    cv2.imshow('image',out)
    cv2.waitKey(0)

if __name__ == "__main__":
    data_dir = "inputs/sample/"
    generateTrueDepth(data_dir)