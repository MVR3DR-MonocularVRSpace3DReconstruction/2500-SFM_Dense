import os
import glob
import open3d as o3d
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from unproject import init_colmap_pointcloud

def read_rgbd_image(color_file, depth_file, depth_trunc=1000, convert_rgb_to_intensity=False):
    color = o3d.io.read_image(color_file)
    depth = o3d.io.read_image(depth_file)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=convert_rgb_to_intensity)
    if True:
        plt.subplot(1, 2, 1)
        plt.title('Redwood grayscale image')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Redwood depth image')
        plt.imshow(rgbd_image.depth)
        plt.show()
    return rgbd_image

def generate_point_cloud(image_dir:str, depth_dir:str, intrinsicM=None, extrinsicM=None, flip=False):
    rgbd = read_rgbd_image(image_dir, depth_dir)
    
    if intrinsicM is None:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.intrinsic_matrix = intrinsicM
    if extrinsicM is None:
        extrinsic = np.identity(4)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic, extrinsic
    )
    if flip:
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd
data_dir = "inputs/lab_v3/"
idx = 0
scene, images, cameras, img_width, img_height = init_colmap_pointcloud(data_dir, False)

image = sorted(glob.glob(data_dir+"depth_completion/outputs/image/*.png"))[idx]
out_depth = sorted(glob.glob(data_dir+"depth_completion/outputs/output_depth/*.png"))[idx]  #    sparse_depth
sparse_depth = sorted(glob.glob(data_dir+"depth_completion/outputs/sparse_depth/*.png"))[idx] 

old_name = Path(sorted(glob.glob(data_dir+"undistorted/images/*.jpg"))[idx]).name

from PIL import Image
tmp_depth = Image.open(out_depth)
tmp_sparse = Image.open(sparse_depth)
print(np.array(tmp_depth), np.array(tmp_sparse))

# input()
pcd = generate_point_cloud(image, out_depth, cameras[images[old_name]["cam_id"]]["intrinsic"])
# pcdo = generate_point_cloud("c.jpg", "a.png", cameras[images["91730.jpg"]["cam_id"]]["intrinsic"])
# pcd.paint_uniform_color([1,0,0])
o3d.visualization.draw_geometries([pcd])
