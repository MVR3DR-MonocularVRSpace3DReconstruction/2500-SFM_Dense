from distutils.command.clean import clean
import os
import sys
import time
import torch
import datetime

import glob
from tqdm import tqdm
from pathlib import Path
import argparse

from unproject import unproject_fusion_mapping
from midas_run import run as midas_run

def run(args):
    times = [0, 0, 0, 0, 0, 0]
    input_dir = args.input
    output_dir = args.output
    data_type = args.image_type
    model_type = args.model_type
    enabled_module = args.pipelines
    toClean = args.clean
    #check openMVS/MVG
    stdout = os.popen("bash checkOpenMVS.sh").read()
    if stdout != "":
        return
    
    # make file clean
    if toClean and int(enabled_module) > 0:
        print("=> You can not run the following procedure after clean up~")
        return
    parent_dir = os.path.basename(os.path.normpath(input_dir))
    if toClean:
        os.system("find {}* | grep -v images | xargs rm -rf".format(input_dir))
        os.system("rm -rf {}/{}".format(output_dir, parent_dir))
    
    Path("{}/{}".format(output_dir, parent_dir)).mkdir(parents=True, exist_ok=True)
    
    # start pipeline
    if int(enabled_module) <= 0:
        start_time = time.time()
        print("\n\n=> OpenSFM constructing...")
        os.system("cp {} {}/config.yaml".format( 
            "config_video_stream.yaml" if data_type == "frames" else "config_disparity.yaml", 
            input_dir))
        os.system("./opensfm_run_all {}".format(input_dir))
        times[0] = time.time() - start_time
        print("====================================")
        print("OpenSFM construction")
        print("====================================")
        print("- OpenSFM construction %s" % datetime.timedelta(seconds=times[0]))
        
    if int(enabled_module) <= 1:
        start_time = time.time()
        print("\n\n=> OpenMVS Densifying...")
        cwd = os.getcwd()
        os.chdir("{}/undistorted/openmvs/".format(input_dir))
        os.system("DensifyPointCloud scene.mvs")
        os.chdir(cwd)
        times[1] = time.time() - start_time
        print("====================================")
        print("OpenMVS Densify")
        print("====================================")
        print("- OpenMVS Densify     %s" % datetime.timedelta(seconds=times[1]))
        
    if int(enabled_module) <= 2:
        start_time = time.time()
        print("\n\n=> Unprojecting...")
        unproject_fusion_mapping(input_dir, False)
        times[2] = time.time() - start_time
        print("====================================")
        print("Unproject")
        print("====================================")
        print("- Unproject           %s" % datetime.timedelta(seconds=times[2]))
        
        
    if int(enabled_module) <= 3:
        start_time = time.time()
        print("\n\n=> Midas Predicting...")
        default_models = {
            "midas_v21_small": "MiDaS/weights/midas_v21_small-70d6b9c8.pt",
            "midas_v21": "MiDaS/weights/midas_v21-f6b98070.pt",
            "dpt_large": "MiDaS/weights/dpt_large-midas-2f21e586.pt",
            "dpt_hybrid": "MiDaS/weights/dpt_hybrid-midas-501f0c75.pt",
        }
        # set torch options
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        Path("{}/relative_depth_predict".format(input_dir)).mkdir(parents=True, exist_ok=True)
        midas_run(input_dir+"/unproject_fusion", 
                input_dir+"/relative_depth_predict", 
                default_models[model_type], 
                model_type, True)
        times[3] = time.time() - start_time
        print("====================================")
        print("Midas Predict")
        print("====================================")
        print("- Midas Predict       %s" % datetime.timedelta(seconds=times[3]))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    
    parser.add_argument("-p", "--pipelines",
                        help="use number 0~4 to select start modules\
                            [0: OpenSFM] [1: OpenMVS] [2: Unproject]\
                            [3: Midas] [4: Patch PointCloud]",
                        choices=[str(idx) for idx in range(5)],
                        default='0')
    
    parser.add_argument("-c", "--clean", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Clean up directory when init second time")
    
    parser.add_argument('-mt', '--model_type', 
        default='dpt_large',
        choices=['midas_v21_small', 'midas_v21', "dpt_large", "dpt_hybrid"],
        help='model type: dpt_large, dpt_hybrid, midas_v21_large or midas_v21_small',
    )
    
    args = parser.parse_args()
    
    run(args)
