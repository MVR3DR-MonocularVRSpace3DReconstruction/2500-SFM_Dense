import numpy
from PIL import Image
import plyfile 
import time
import math

import os
import json
import cv2
import open3d as o3d
import open3d.visualization as vis
import numpy as np
import rtvec2extrinsic

print('Importing')
now = time.time()
pcd_path = '../data/LAB410/undistorted/openmvs/scene_dense.ply'
plydata = plyfile.PlyData.read(pcd_path)


print('Imported in',time.time()-now,"seconds")
vertex_list=plydata.elements[0]
print(plydata.elements[0])
x =vertex_list["x"]
# swapped z and y as they seem to be using a different convention...
y =vertex_list["y"]
z =vertex_list["z"]
red = vertex_list['red']
green = vertex_list['green']
blue = vertex_list['blue']
xyz1 = np.c_[x,y,z,np.ones(len(x))]
rgb = np.c_[red,green,blue]


R=[rx,ry,rz]=[ 1.22549060388722,-1.2841011616291398, 1.2869643307339165] 
T=[tx,ty,tz]=[ -6.145350049497323, -0.983444215613622, 20.00074193758397]

input()

M=cv2.Rodrigues(numpy.array(R))[0]
M=numpy.concatenate([M,numpy.matrix(T).T],axis=1)
M=numpy.concatenate([M,numpy.matrix([0,0,0,1])],axis=0)

xyz_c=(M*(xyz1.T)).T
x=numpy.array(xyz_c[:,0]).reshape(len(xyz_c))
y=numpy.array(xyz_c[:,1]).reshape(len(xyz_c))
z=numpy.array(xyz_c[:,2]).reshape(len(xyz_c))

# convert to polar coords
start = time.time()
rho = np.sqrt(x**2+y**2) #cylindrical radius
phi = np.arctan2(y,x) #azimuthal angle
phi_mod = np.mod(phi,2*np.pi)
r = np.sqrt(x**2+y**2+z**2) #spherical angle (x,y)
theta = np.arccos(z/r) #polar angle (z)
end = time.time()
print('Converted in {:.3} seconds'.format(end-start))

# projection
width=2364
height=1773
image_cloud=numpy.zeros((height,width,3),dtype=numpy.uint8)

# generate x and y coordinates for image (round with cast to int)
im_y = (theta/np.pi*(height-1)).astype(np.int32)
im_x = (np.mod(phi,2*np.pi)/(2*np.pi)*(width-1)).astype(np.int32)
image_cloud[im_y,im_x] = rgb

output_folder="./"
os.makedirs(output_folder,exist_ok=True)
Image.fromarray(image_cloud).save(os.path.join(output_folder,"reprojected-cloud.png"))