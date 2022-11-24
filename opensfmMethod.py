# import sys
# sys.path.insert(0,'..')
from OpenSfM.opensfm import features, pygeometry, dataset
import cv2
import numpy as np
import open3d as o3d

base_path = "inputs/lab_v3"
# load
data = dataset.DataSet(base_path)
rec = data.load_reconstruction()[0]
# run the projections

shot = rec.shots["IMG20221123181830.jpg"]

# project
im  = data.load_image("IMG20221123181830.jpg")
h, w, _ = im.shape
cam = shot.camera
for pt in rec.points.values():
    pt2D = shot.project(pt.get_global_pos())
    pt2D_px = cam.normalized_to_pixel_coordinates(pt2D)
    if pt2D_px[0] >= 0 and pt2D_px[0] < w and pt2D_px[1] >= 0 and pt2D_px[1] < h:
        cv2.circle(im, (int(pt2D_px[0]), int(pt2D_px[1])), 1, (255, 0, 0), 1)

cv2.imwrite(base_path + "/img2.jpg", im)
im  = data.load_image("IMG20221123181830.jpg")
nube = o3d.io.read_point_cloud(base_path + "/merged.ply")
for pt3D in nube.points:
    pt2D = shot.project(pt3D)
    pt2D_px = cam.normalized_to_pixel_coordinates(pt2D)
    if pt2D_px[0] >= 0 and pt2D_px[0] < w and pt2D_px[1] >= 0 and pt2D_px[1] < h:
        cv2.circle(im, (int(pt2D_px[0]), int(pt2D_px[1])), 1, (255, 0, 0), 1)
cv2.imwrite(base_path + "/img3.jpg", im)
