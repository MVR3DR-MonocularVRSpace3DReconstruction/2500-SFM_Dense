
import glob
import numpy as np
from copy import deepcopy
from pathlib import Path

import open3d as o3d
import pycolmap

import cv2
from PIL import Image

import json
###############################################
# Read Colmap & Point cloud
###############################################

def init_colmap_pointcloud(data_dir, debug = False):
    """read colmap file and densitied pointcloud return dictionary

    Args:
        data_dir (str): path of inputs
        debug (bool, optional): visualize generated pointcloud and cameras. Defaults to False.

    Returns:
        _type_: _description_
    """
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

    # print("Intrinsic:\n", cameras[1]["intrinsic"])
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
        dict_image["id"] = image.image_id
        dict_image["lineset"] = cam
        dict_image["cam_id"] = image.camera_id
        images[image.name] = dict_image
        if debug:
            camera_linesets.append(cam)
    if debug:
        print("=> Images:\n", images)
        o3d.visualization.draw_geometries([pcd, *camera_linesets])
        
    return pcd, images, cameras, img_width, img_height

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

def unproject_fusion_mapping(data_dir, debug):
    """unproject the depth map from pointcloud and colmap camera path

    Args:
        data_dir (str): input data path
        debug (bool): show debug options
    
    Returns:
        depth map in data_dir/unproject_depth/
        color map in data_dir/unproject_colors/
    """
    # fusion_dir = Path("{}/unproject_fusion".format(data_dir))
    depth_dir = Path("{}/unproject_depth".format(data_dir))
    unproject_dir = Path("{}/unproject_colors".format(data_dir))
    intrinsic_dir = Path("{}/camera_intrinsic".format(data_dir))
    # fusion_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    unproject_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_dir.mkdir(parents=True, exist_ok=True)

    images_dir = sorted(glob.glob(data_dir+"/undistorted/images/*.jpg"))
    
    pcd, images, cameras, img_width, img_height = init_colmap_pointcloud(data_dir, debug)
    # export cameras data
    # print(cameras)
    for camera in cameras.keys():
        # print(camera, cameras[camera]["intrinsic"])
        with open( str(intrinsic_dir/(str(camera)+'.npy')), 'wb') as f:
            np.save(f, cameras[camera]["intrinsic"])
    # create 3d renderer
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    renderer_pc = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
    renderer_pc.scene.set_background(np.array([0, 0, 0, 0]))
    renderer_pc.scene.add_geometry("pcd", pcd, mat)
    for img_path in images_dir:
        
        img_name_ext = img_path.split("/")[-1]
        print("=> Load image: {}".format(img_name_ext))
        renderer_pc.setup_camera(
            cameras[images[img_name_ext]["cam_id"]]["intrinsic"], 
            np.linalg.inv(images[img_name_ext]["extrinsic"]), 
            img_width, img_height)
        
        # depth_image = np.asarray(renderer_pc.render_to_depth_image(True))
        # color_image = np.asarray(renderer_pc.render_to_image())[:,:,::-1]
        depth_image = renderer_pc.render_to_depth_image(True)
        depth_image = np.array(depth_image)
        
        color_image = renderer_pc.render_to_image()
        img_name = img_name_ext.split(".")[0]
        Image.fromarray(depth_image*255).convert("I").save(str(depth_dir/"{}.png".format(img_name)))
        # o3d.io.write_image(str(depth_dir/"{}.png".format(img_name)), depth_image)
        o3d.io.write_image(str(unproject_dir/"{}.jpg".format(img_name)), color_image)
        if debug:
            print(np.array(depth_image))
        # np.save(str(depth_dir/"depth_{}".format(img_name)), depth_image)
        # cv2.imwrite(str(depth_dir/"{}.png".format(img_name)), depth_image)
        # cv2.imwrite(str(unproject_dir/"{}.png".format(img_name)), color_image)

        # origin_image = cv2.imread(data_dir+"images/"+img_name_ext)
        # overlay_image = dark2alpha(color_image)
        # fusion_image = merge_image(origin_image, overlay_image, 0, 0)

        # cv2.imwrite(str(fusion_dir/"fusion_{}.png".format(img_name)), fusion_image)
    return pcd, images, cameras
###############################################
# Back to Point cloud
###############################################



if __name__ == "__main__":
    data_dir = "inputs/lab_v3/"
    images_name = Path(sorted(glob.glob(data_dir+"undistorted/images/*.jpg"))[0]).name
    
    pcd, images, cameras, img_width, img_height = init_colmap_pointcloud(data_dir, True)
    print(img_width, img_height)
    # img_width = 1600
    # img_height = 1200
    print("Image file: ", images_name)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    print(cameras[images[images_name]["cam_id"]]["intrinsic"])

    intri = np.array([[3000,0,2000],
                      [0,3000,1500],
                      [0,0,1]])

    renderer_pc = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
    renderer_pc.scene.set_background(np.array([0, 0, 0, 0]))
    renderer_pc.scene.add_geometry("pcd", pcd, mat)
    renderer_pc.setup_camera(
        # cameras[images[images_name]["cam_id"]]["intrinsic"], 
        intri,
        np.linalg.inv(images[images_name]["extrinsic"]), 
        img_width, img_height)

    depth_image = renderer_pc.render_to_depth_image(True)
    color_image = renderer_pc.render_to_image()
    depth_image = np.array(depth_image)
    Image.fromarray(depth_image*255).convert("I").save("depth.png")
    # o3d.io.write_image("depth.png", depth_image)
    o3d.io.write_image("color.png", color_image)

    color_img = Image.open(sorted(glob.glob(data_dir+"images/*.jpg"))[0])
    import matplotlib.pyplot as plt
    plt.subplot(1, 3, 1)
    plt.title(' color image')
    plt.imshow(color_img)
    plt.subplot(1, 3, 2)
    plt.title(' unproject color image')
    color = plt.imread("color.png")
    depth = plt.imread("depth.png")
    plt.imshow(color)
    plt.subplot(1, 3, 3)
    plt.title(' depth image')
    plt.imshow(depth)
    plt.show()