


import cv2
import os
import json
import math
import numpy as np
from copy import deepcopy
import open3d as o3d
import open3d.visualization as vis
import pycolmap

# import matplotlib.pyplot as plt
from PIL import Image
import rtvec2extrinsic

debug = True
###############################################
# Read Colmap & Point cloud
###############################################
data_dir = "data/LAB410/"
pcd_path = data_dir+"undistorted/openmvs/scene_dense.ply"
colmap_path = data_dir+"colmap_export"
pcd = o3d.io.read_point_cloud(pcd_path)
colmap = pycolmap.Reconstruction(colmap_path)
print(colmap.summary())

img_width, img_height = (-1, -1)
cameras = {}
for camera in colmap.cameras.values():
    # print(camera.camera_id)
    dict_camera = {}
    dict_camera["lineset"] = o3d.geometry.LineSet.create_camera_visualization(
        camera.width, camera.height, camera.calibration_matrix(), np.identity(4), scale=1)
    dict_camera["intrinsic"] = camera.calibration_matrix()
    cameras[camera.camera_id] = dict_camera
    img_width, img_height = (camera.width, camera.height)

print("Intrinsic:\n", cameras[1]["intrinsic"])
images = {}
if debug:
    camera_linesets = []
for image in colmap.images.values():
    dict_image = {}
    
    T = np.eye(4)
    T[:3, :4] = image.inverse_projection_matrix()
    
    cam = deepcopy(cameras[image.camera_id]["lineset"]).transform(T)
    cam.paint_uniform_color([1.0, 0.0, 0.0])  # red
    
    dict_image["extrinsic"] = T
    dict_image["name"] = image.name
    dict_image["lineset"] = cam
    dict_image["cam_id"] = image.camera_id
    images[image.image_id] = dict_image
    if debug:
        camera_linesets.append(cam)
if debug:
    vis.draw_geometries([pcd, *camera_linesets])

###############################################
# Render Depth & Colors
###############################################
def merge_image(back, front, x,y):
    # convert to rgba
    if back.shape[2] == 3:
        back = cv2.cvtColor(back, cv2.COLOR_BGR2BGRA)
    if front.shape[2] == 3:
        front = cv2.cvtColor(front, cv2.COLOR_BGR2BGRA)

    # crop the overlay from both images
    bh,bw = back.shape[:2]
    fh,fw = front.shape[:2]
    x1, x2 = max(x, 0), min(x+fw, bw)
    y1, y2 = max(y, 0), min(y+fh, bh)
    front_cropped = front[y1-y:y2-y, x1-x:x2-x]
    back_cropped = back[y1:y2, x1:x2]

    alpha_front = front_cropped[:,:,3:4] / 255
    alpha_back = back_cropped[:,:,3:4] / 255
    
    # replace an area in result with overlay
    result = back.copy()
    # print(f'af: {alpha_front.shape}\nab: {alpha_back.shape}\nfront_cropped: {front_cropped.shape}\nback_cropped: {back_cropped.shape}')
    result[y1:y2, x1:x2, :3] = alpha_front * front_cropped[:,:,:3] + (1-alpha_front) * back_cropped[:,:,:3]
    result[y1:y2, x1:x2, 3:4] = (alpha_front + alpha_back) / (1 + alpha_front*alpha_back) * 255

    return result


def dark2alpha(img):
    tmp_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp_img, 5, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha]
    img_out = cv2.merge(rgba, 4)
    return img_out

image_idx = 35
print("Image file: ", images[image_idx]["name"])
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = 'defaultUnlit'

renderer_pc = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
renderer_pc.scene.set_background(np.array([0, 0, 0, 0]))
renderer_pc.scene.add_geometry("pcd", pcd, mat)
renderer_pc.setup_camera(
    cameras[images[image_idx]["cam_id"]]["intrinsic"], 
    np.linalg.inv(images[image_idx]["extrinsic"]), 
    img_width, img_height)

depth_image = np.asarray(renderer_pc.render_to_depth_image())
color_image = np.asarray(renderer_pc.render_to_image())[:,:,::-1]
depth_view = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255
# depth_image = depth_image * 255
np.save('depth', depth_image)
cv2.imwrite("depth.png", depth_image)
cv2.imwrite("depth_view.png", depth_view)
cv2.imwrite("color.png", color_image)

origin_image = cv2.imread(data_dir+"images/"+images[image_idx]["name"])
overlay_image = dark2alpha(color_image)
fusion_image = merge_image(origin_image, overlay_image, 0, 0)

cv2.imwrite("fusion.png", fusion_image)

###############################################
# Back to Point cloud
###############################################

