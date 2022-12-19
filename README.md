# Structure From Motion

---------------

## Introduction

本專案主要用於自動化建制稠密化點雲模型

-   輸入：圖片（推薦）、視頻
-   輸出：稠密化點雲、特征圖、投影深度圖、投影色彩圖、補充深度圖（如果有配置mondi-depth的話）
-   

## Configuration

-   Ubuntu 18.04 or higher
-   python 3.6.9 (其他版本未經測試)

## Install

```
git clone https://github.com/MVR3DR-MonocularVRSpace3DReconstruction/2500-SFM_Dense.git
cd 2500-SFM_Dense
```

### OpenSfm

在 2500-SFM_Dense 文檔下，或者可以在其他路徑安裝並複製到該文檔下。
克隆文檔並確認版本

```
git clone --recursive https://github.com/mapillary/OpenSfM
cd OpenSfM && git checkout v0.5.1
```

安裝依賴

```
sudo apt-get install build-essential cmake libatlas-base-dev libatlas-base-dev libgoogle-glog-dev \
libopencv-dev libsuitesparse-dev python3-pip python3-dev  python3-numpy python3-opencv \
python3-pyproj python3-scipy python3-yaml libeigen3-dev
```

#### opengv

```
cd ..
mkdir source && cd source/
git clone --recurse-submodules -j8 https://github.com/laurentkneip/opengv.git
cd opengv && mkdir build && cd build
cmake .. -DBUILD_TESTS=OFF -DBUILD_PYTHON=ON -DPYBIND11_PYTHON_VERSION=3.6 -DPYTHON_INSTALL_DIR=/usr/local/lib/python3.6/dist-packages/
sudo make install
```

#### ceres

```
cd ../../
curl -L http://ceres-solver.org/ceres-solver-1.14.0.tar.gz | tar xz
cd ./ceres-solver-1.14.0 && mkdir build-code && cd build-code
cmake .. -DCMAKE_C_FLAGS=-fPIC -DCMAKE_CXX_FLAGS=-fPIC -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
sudo make -j8 install
```

安裝OpenSfm依賴

```
cd OpenSfm
sudo apt-get update \
    && sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        libeigen3-dev \
        libopencv-dev \
        libceres-dev \
        python3-dev \
        python3-numpy \
        python3-opencv \
        python3-pip \
        python3-pyproj \
        python3-scipy \
        python3-yaml \
        curl
pip3 install -r requirements
python3 setup.py build
```

註：如果build失敗建議移除cmake_build 文檔再嘗試其他操作

### OpenMVS

```
#Prepare and empty machine for building:
sudo apt-get update -qq && sudo apt-get install -qq
sudo apt-get -y install git cmake libpng-dev libjpeg-dev libtiff-dev libglu1-mesa-dev
main_path=`pwd`

#Eigen (Required)
git clone https://gitlab.com/libeigen/eigen.git --branch 3.4
mkdir eigen_build && cd eigen_build
cmake . ../eigen
make && sudo make install
cd ..

#Boost (Required)
sudo apt-get -y install libboost-iostreams-dev libboost-program-options-dev libboost-system-dev libboost-serialization-dev

#OpenCV (Required)
sudo apt-get -y install libopencv-dev

#CGAL (Required)
sudo apt-get -y install libcgal-dev libcgal-qt5-dev

#VCGLib (Required)
git clone https://github.com/cdcseacave/VCG.git vcglib

#Ceres (Optional)
sudo apt-get -y install libatlas-base-dev libsuitesparse-dev
git clone https://ceres-solver.googlesource.com/ceres-solver ceres-solver
mkdir ceres_build && cd ceres_build
cmake . ../ceres-solver/ -DMINIGLOG=ON -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
make -j2 && sudo make install
cd ..

#GLFW3 (Optional)
sudo apt-get -y install freeglut3-dev libglew-dev libglfw3-dev

#OpenMVS
git clone https://github.com/cdcseacave/openMVS.git openMVS
cd openMVS && git checkout v2.0.1 && cd .. # 新增返回到對應版本
mkdir openMVS_build && cd openMVS_build
cmake . ../openMVS -DCMAKE_BUILD_TYPE=Release -DVCG_ROOT="$main_path/vcglib"

#If you want to use OpenMVS as shared library, add to the CMake command:
-DBUILD_SHARED_LIBS=ON

#Install OpenMVS library (optional):
make -j2 && sudo make install
```

安裝成功後，找到 « openMVS_build » 路徑
並將 « openMVS_build/bin » 添加到環境的PATH中

### mondi-python

如果需要繼續開發後續稠密點雲完善（對投影的深度圖做depth completion）
請參照[alexklwong/mondi-python (github.com)](https://github.com/alexklwong/mondi-python) 安裝環境，並將 mondi-python 的文檔路徑連接到 « 2500-SFM_Dense » 的根目錄下

## Run

1.   將拍攝的照片資訊放入« ./inputs/<自定義命名>/images/ » 中
     -   如果需要配置opensfm 運行參數可編輯根目錄下 config_disparity.yaml // config_video_stream.yaml文檔
     -   相機焦距參數 修改根目錄下 camera_models.json
2.   在專案根目錄下執行 >>> python3 main.py -i <文檔路徑 inputs/<自定義命名>/ > -c
3.   等待程式完成

## Data Structure

當Run 全部完成後，資料結構：

```
camera_intrinsic					# 相機內參矩陣文檔
	- *.npy		# 相機內參矩陣參數
colmap_export						# pycolmap 輸出相機及sfm資料
	- cameras.txt
	- colmap_database.db
	- images.txt
	- points3D.txt
	- project.ini
depth_completion					# mondi-python 執行輸出結果
	- outputs		# depth completion 輸出
		- ground_truth
		- image
		- output_depth
		- sparse_depth
	- result.txt	# depth completion 記錄
exif								# 圖片提取exif文檔
	- *.exif
features							# 圖片提取特征點文檔
	- *.features.npz
		- colors.npy
		- descriptors.npy
		- instances.npy
		- OPENSFM_FEATURES_VERSION.npy
		- points.npy
		- segementation_labels.npy
		- segmentations.npy
images								# 輸入圖片
	- *		# 一般放入.png .jpg 文檔
matches								# 對應圖片特征點
	- *.matches.pkl.gz
reports								# OpenMVS 報告
	- features
		- *.json
	- features.json
	- matches.json
	- reconstruction.json
	- tracks.json
undistorted							# 輸出SFM點雲基本文檔
	- depthmaps		# 深度資訊
		- *.clean.npz
			- depth.npy
			- plane.npy
			- score.npy
		- *.pruned.npz
		- merged.ply	# << 輸出包含大量雜點的點雲 ply 格式 >>
	- images		# 引用到的圖片副本
		- *		#依據使用者輸入
	- openmvs		# openmvs格式輸出
		- *.dmap	# 深度圖資訊
		- scene.mvs	# << 輸出包含大量雜點的點雲 openmvs 格式 >>
		- scene_dense.mvs # << 輸出去除雜點的點雲 openmvs 格式 >>
		- scene_dense.ply # << 輸出去除雜點的點雲 ply 格式 >>
unproject_colors					# 依據scene_dense.ply 渲染的色彩圖
	- *.jpg		# 色彩圖
unproject_depth						# 依據scene_dense.ply 渲染的深度圖
	- *.png		# 深度圖
camera_models.json					# 預估相機模型參數
camera_models_overrides.json		# 預設相機模型參數
config.yaml							# 配置文件
custom_depth.txt					# 自定義深度文件
custom_image.txt					# 自定義圖片文件
custom_intrinsic.txt				# 自定義相機內參
profile.log						
reconstruction.json					# SFM重建畸變參數
reconstruction.meshed.json
reference_lla.json
tracks.csv							# 相機坐標
```

## References

OpenMVS： [cdcseacave/openMVS: open Multi-View Stereo reconstruction library (github.com)](https://github.com/cdcseacave/openMVS) 

OpenSfm： [mapillary/OpenSfM: Open source Structure-from-Motion pipeline (github.com)](https://github.com/mapillary/OpenSfM)

Monitored Distillation for Positive Congruent Depth Completion (MonDi)：[alexklwong/mondi-python (github.com)](https://github.com/alexklwong/mondi-python)

