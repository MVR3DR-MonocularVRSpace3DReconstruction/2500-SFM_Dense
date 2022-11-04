import os
import glob
import cv2
from PIL import Image, ImageChops
import numpy as np
from pathlib import Path
import copy

###############################################
# Block Sample Method
###############################################

def blockSampleFusionByColor(estimate_block, o3d_block, debug = False):
    if debug:
        print(estimate_block, "\n", o3d_block)
    assert estimate_block.size == o3d_block.size
    print(o3d_block.size)
    output_block = copy.deepcopy(o3d_block)
    # get valid pixel in open3d block
    valid_pix = np.where(o3d_block != 0)
    # get the value of mapping valid pixel
    omapping_pix = o3d_block[valid_pix]
    emapping_pix = estimate_block[valid_pix]

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
        if debug:
            print("output block:\n", output_block)
        
        # cv2.imshow('image',output_block)
        # cv2.waitKey(0)

        # print(output_block)
        # input()
    return output_block
    
    
def generateTrueDepthByBlockSample(data_dir, sizeof_sample_block = 32, valid_threshold = 0.3, debug = False):
    """fusion the original depth and estimate depth block by block

    Args:
        data_dir (str): inputs data path
        sizeof_sample_block (int, optional): sample block size for nxn fetched from input image. Defaults to 32.
        valid_threshold (float, optional): available block threshold, take the ratio of valid original depth. Defaults to 0.3.
        debug (bool, optional): debug~. Defaults to False.
        
    Returns:
        fused true depth map - uint32 png
    """
    estimate_depth_dir = data_dir + "relative_depth_predict/"
    o3d_depth_dir = data_dir + "unproject_depth/"
    
    edepth_path = sorted(glob.glob(estimate_depth_dir+"*.png"))
    odepth_path = sorted(glob.glob(o3d_depth_dir+ "*.png"))
    
    file_idx = 0
    edepth_seq = Path(edepth_path[file_idx]).stem.split("_")[-1]
    odepth_seq = Path(odepth_path[file_idx]).stem.split("_")[-1]
    assert edepth_seq == odepth_seq

    edepth = cv2.imread(edepth_path[file_idx], cv2.IMREAD_UNCHANGED) # , cv2.IMREAD_GRAYSCALE
    edepth = cv2.bitwise_not(edepth)
    odepth = cv2.imread(odepth_path[file_idx], cv2.IMREAD_UNCHANGED) # , cv2.IMREAD_GRAYSCALE

    if debug:
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image',edepth)
        cv2.waitKey(0)
        cv2.imshow('image',odepth)
        cv2.waitKey(0)

        print("estimate depth: \n", edepth,"\n\noriginal depth: \n", odepth)
        print("\nestimate depth type: {}, min: {}, max: {}".format(edepth.dtype, np.min(edepth), np.max(edepth)))
        print("origin depth type: {}, min: {}, max: {}".format(odepth.dtype, np.min(odepth), np.max(odepth)))
        print("="*50+"\n\n")

    # print(edepth,"\n", odepth)
    assert edepth.shape == odepth.shape
    out = copy.deepcopy(odepth)
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
                merged_sample_block = blockSampleFusionByColor(esample_block, osample_block, debug)
                out[pix_col:min(edepth.shape[0], pix_col+sizeof_sample_block), \
                    pix_row:min(edepth.shape[1], pix_row+sizeof_sample_block)] = merged_sample_block
                if debug:
                    cv2.imshow('image',out)
                    cv2.waitKey(0)
            else:
                out[pix_col:min(edepth.shape[0], pix_col+sizeof_sample_block), \
                    pix_row:min(edepth.shape[1], pix_row+sizeof_sample_block)] = osample_block
            
    print("=> Total {} blocks, {} valid blocks, {:.3}% of Image Mended".format(
        block_counter, valid_block_counter, valid_block_counter/block_counter*100))

    out = Image.fromarray(out).convert("I").save("d.png")


###############################################
# Total merged Method
###############################################

def merge_image(back, front):
    mask = np.where(front != 0)
    back[mask] = front[mask]
    return back


def blockSampleFusionByRatio(estimate_block, o3d_block, debug = False):
    if debug:
        print("e:\n", estimate_block, "\no:\n", o3d_block)
        print("size:",o3d_block.size)
    assert estimate_block.size == o3d_block.size
    
    # get valid pixel in open3d block
    valid_pix = np.where(o3d_block != 0)
    # get the value of mapping valid pixel
    omapping = o3d_block[valid_pix]
    emapping = estimate_block[valid_pix]
    
    divide = omapping / emapping
    divide = divide[divide != np.inf]
    if len(divide) == 0:
        return o3d_block
    e2o_ratio = np.average(divide)
    
    output_block = (estimate_block * e2o_ratio).astype(np.uint16)
    if debug:
        print("estimate to original ratio: ", e2o_ratio)
        print("estimate mapping pixel:\n", emapping, "\noriginal mapping pixel:\n", omapping)
        print("estimate min: {}, max: {}".format(np.min(emapping), np.max(emapping)))
        print("origin min: {}, max: {}".format(np.min(omapping), np.max(omapping)))
        print("Out image:\n", output_block)

    output_block = merge_image(output_block, o3d_block)
    return output_block
    
def generateTrueDepthByTotalMerge(data_dir, sizeof_sample_block = 128, valid_threshold = 0.2, debug = False):
    estimate_depth_dir = data_dir + "relative_depth_predict/"
    o3d_depth_dir = data_dir + "unproject_depth/"
    
    edepth_path = sorted(glob.glob(estimate_depth_dir+"*.png"))
    odepth_path = sorted(glob.glob(o3d_depth_dir+ "*.png"))
    
    file_idx = 35
    print("=> Mending depth {}".format(odepth_path[file_idx]))
    edepth_seq = Path(edepth_path[file_idx]).stem.split("_")[-1]
    odepth_seq = Path(odepth_path[file_idx]).stem.split("_")[-1]
    assert edepth_seq == odepth_seq

    edepth = cv2.imread(edepth_path[file_idx], cv2.IMREAD_GRAYSCALE) # , cv2.IMREAD_GRAYSCALE
    edepth = cv2.bitwise_not(edepth)
    odepth = cv2.imread(odepth_path[file_idx], cv2.IMREAD_UNCHANGED) # , cv2.IMREAD_GRAYSCALE
    
    if debug:
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image',edepth)
        cv2.waitKey(0)
        cv2.imshow('image',odepth)
        cv2.waitKey(0)

        print("estimate depth: \n", edepth,"\n\noriginal depth: \n", odepth)
        print("\nestimate depth type: {}, min: {}, max: {}".format(edepth.dtype, np.min(edepth), np.max(edepth)))
        print("origin depth type: {}, min: {}, max: {}".format(odepth.dtype, np.min(odepth), np.max(odepth)))
        print("="*50+"\n\n")
    assert edepth.shape == odepth.shape
    edepth = np.array(edepth, dtype = "uint16") * 255 
    
    valid_pix = np.where(odepth != 0)
    omap = odepth[valid_pix]
    emap = edepth[valid_pix]
    odepthmap_range = [np.min(odepth), np.max(odepth)]
    edepthmap_range = [np.min(edepth), np.max(edepth)]
    edepth = ((edepth - edepthmap_range[0]) * ((odepthmap_range[1]-odepthmap_range[0])/(edepthmap_range[1]-edepthmap_range[0]))).astype(np.uint16)
    if debug:
        print(edepth)
        cv2.imshow('image',edepth)
        cv2.waitKey(0)

    # print(odepth_range)
    out = copy.deepcopy(odepth)
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
                merged_sample_block = blockSampleFusionByRatio(esample_block, osample_block, debug)
                out[pix_col:min(edepth.shape[0], pix_col+sizeof_sample_block), \
                    pix_row:min(edepth.shape[1], pix_row+sizeof_sample_block)] = merged_sample_block
                if debug:
                    cv2.imshow('image',out)
                    cv2.waitKey(0)
            else:
                out[pix_col:min(edepth.shape[0], pix_col+sizeof_sample_block), \
                    pix_row:min(edepth.shape[1], pix_row+sizeof_sample_block)] = osample_block
            
    out = Image.fromarray(out).convert("I")
    out.save("d.png")
    
    
###############################################
# Edge detection Method
###############################################
  
def generateTrueDepthByEdgeDetect(data_dir, valid_point_threshold = 10, debug = False):
    estimate_depth_dir = data_dir + "relative_depth_predict/"
    o3d_depth_dir = data_dir + "unproject_depth/"
    image_dir = data_dir + "undistorted/images/"
    
    edepth_path = sorted(glob.glob(estimate_depth_dir+"*.png"))
    odepth_path = sorted(glob.glob(o3d_depth_dir+ "*.png"))
    image_path = sorted(glob.glob(image_dir+"*.jpg"))
    
    file_idx = 0
    edepth_seq = Path(edepth_path[file_idx]).stem.split("_")[-1]
    odepth_seq = Path(odepth_path[file_idx]).stem.split("_")[-1]
    image_seq = Path(image_path[file_idx]).stem
    assert edepth_seq == odepth_seq == image_seq

    edepth = cv2.imread(edepth_path[file_idx], cv2.IMREAD_GRAYSCALE) # , cv2.IMREAD_GRAYSCALE
    edepth = cv2.bitwise_not(edepth)
    odepth = cv2.imread(odepth_path[file_idx], cv2.IMREAD_UNCHANGED) # , cv2.IMREAD_GRAYSCALE
    image = cv2.imread(image_path[file_idx], cv2.IMREAD_UNCHANGED)
    
    if debug:
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image',edepth)
        cv2.waitKey(0)
        cv2.imshow('image',odepth)
        cv2.waitKey(0)

        print("estimate depth: \n", edepth,"\n\noriginal depth: \n", odepth)
        print("\nestimate depth type: {}, min: {}, max: {}".format(edepth.dtype, np.min(edepth), np.max(edepth)))
        print("origin depth type: {}, min: {}, max: {}".format(odepth.dtype, np.min(odepth), np.max(odepth)))
        print("="*50+"\n\n")
    assert edepth.shape == odepth.shape
    
    copy_edepth = np.uint8(edepth)
    edepth = np.array(edepth, dtype = "uint16") * 255 
    planes = np.load("m.npy")
    # print(planes)
    for plane in planes:
        plane = np.squeeze(plane, axis=(2,)).astype(np.uint8) * 255
        plane = cv2.resize(plane, (odepth.shape[1], odepth.shape[0]), interpolation=cv2.INTER_AREA) # , 
        
        mask = np.where(plane != 255)
        # Image.fromarray(plane).show()
        odepth_slice = copy.deepcopy(odepth)
        odepth_slice[mask] = 0
        
        if debug:
            cv2.imshow('image',plane)
            cv2.waitKey(0)
            cv2.imshow('image',odepth_slice)
            cv2.waitKey(0)
        
            image_slice = copy.deepcopy(image)
            image_slice[mask] = 0
            cv2.imshow('image',image_slice)
            cv2.waitKey(0)
            print(odepth_slice[odepth_slice != 0])
            
        if len(odepth_slice[odepth_slice != 0]) < valid_point_threshold:
            print("=> odepth_slice do not have enought points:\n", odepth_slice[odepth_slice != 0])
            continue

        

    
    out = copy.deepcopy(odepth)
    
    out = Image.fromarray(out).convert("I")
    out.save("d.png")
    

if __name__ == "__main__":
    data_dir = "inputs/sample/"
    generateTrueDepthByEdgeDetect(data_dir, debug=True)