import os
import sys
import time
import torch
import datetime

import glob
from tqdm import tqdm
from pathlib import Path
import argparse

from midas_run import run as midas_run

def run(input_dir, output_dir, data_type, model_type):
    times = [0, 0, 0, 0, 0, 0]
    
    #check openMVS/MVG
    stdout = os.popen("bash checkOpenMVS.sh").read()
    if stdout != "":
        return
    
    ldir_name = os.path.basename(os.path.normpath(input_dir))
    Path("{}/{}".format(output_dir, ldir_name)).mkdir(parents=True, exist_ok=True)
    
    os.system("cp {} {}/config.yaml".format( 
        "config_video_stream.yaml" if data_type == "frames" else "config_disparity.yaml", 
        input_dir))
    
    start_time = time.time()
    print("=> OpenSFM constructing...")
    os.system("./opensfm_run_all {}".format(input_dir))
    times[0] = time.time() - start_time
    print("====================================")
    print("OpenSFM construction")
    print("====================================")
    print("- OpenSFM construction %s" % datetime.timedelta(seconds=times[0]))
    
    
    start_time = time.time()
    print("=> OpenMVS Densifying...")
    cwd = os.getcwd()
    os.chdir("{}/undistorted/openmvs/".format(input_dir))
    os.system("DensifyPointCloud scene.mvs")
    os.chdir(cwd)
    times[1] = time.time() - start_time
    print("====================================")
    print("OpenMVS Densify")
    print("====================================")
    print("- OpenMVS Densify     %s" % datetime.timedelta(seconds=times[1]))
    
    
    
    start_time = time.time()
    print("=> Midas Predicting...")
    Path("{}/relative_depth_predict".format(input_dir)).mkdir(parents=True, exist_ok=True)
    times[2] = time.time() - start_time
    print("====================================")
    print("Midas Predict")
    print("====================================")
    print("- Midas Predict       %s" % datetime.timedelta(seconds=times[2]))
    
    
    
    start_time = time.time()
    default_models = {
        "midas_v21_small": "MiDaS/weights/midas_v21_small-70d6b9c8.pt",
        "midas_v21": "MiDaS/weights/midas_v21-f6b98070.pt",
        "dpt_large": "MiDaS/weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "MiDaS/weights/dpt_hybrid-midas-501f0c75.pt",
    }

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # midas_run("{}/{}/undistroted/")
    
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFM")
    parser.add_argument("-i", "--input",
                        help="path to the inputs directory",
                        default="inputs/sample/")
    parser.add_argument("-o", "--output",
                        help="path to the outputs directory",
                        default="outputs/")

    parser.add_argument("-t", "--image_type",
                        help="select file type",
                        choices=['frames', 'disparity'],
                        default="disparity")
    
    parser.add_argument('-mt', '--model_type', 
        default='dpt_large',
        choices=['midas_v21_small', 'midas_v21', "dpt_large", "dpt_hybrid"],
        help='model type: dpt_large, dpt_hybrid, midas_v21_large or midas_v21_small',
    )
    
    args = parser.parse_args()
    
    run(args.input, args.output, args.image_type, args.model_type)
