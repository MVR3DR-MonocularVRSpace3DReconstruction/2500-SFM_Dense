import os
import glob
from pathlib import Path

from unproject import init_colmap_pointcloud
def createPathFile(data_dir):
    data_dir = Path(data_dir)
    images_path = data_dir/"undistorted"/"images"
    depth_path = data_dir/"unproject_depth"
    intrinsic_path = data_dir/"camera_intrinsic"

    images = sorted(glob.glob(str(images_path)+"/*.jpg"))
    images = [image.replace("/undistorted/","/") for image in images]
    depth_maps = sorted(glob.glob(str(depth_path)+"/*.png"))
    # intrinsics = sorted(glob.glob(str()+"/*.npy"))

    assert len(images) == len(depth_maps)

    f = open(str(data_dir)+"/custom_image.txt", 'w')
    f.write("\n".join(images))
    f.close()

    f = open(str(data_dir)+"/custom_depth.txt", 'w')
    f.write("\n".join(depth_maps))
    f.close()

    _, images_dict, _, _, _ = init_colmap_pointcloud(str(data_dir)+"/", False)
    # print(images_dict)
    f = open(str(data_dir)+"/custom_intrinsic.txt", 'w')
    for img in images:
        # print(str(intrinsic_path.parents[0])+"/"+str(images_dict[Path(img).name]["cam_id"])+".npy")
        f.write(str(intrinsic_path)+"/"+str(images_dict[Path(img).name]["cam_id"])+".npy\n")
    f.close()

    
if __name__ == "__main__":
    createPathFile("inputs/lab/")