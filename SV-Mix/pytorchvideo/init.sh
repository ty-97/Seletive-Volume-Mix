#!/bin/bash

# compile custom operators
rm pytorchvideo/functional/cuda_sifa/*.so
rm -r pytorchvideo/functional/cuda_sifa/build
rm -r pytorchvideo/functional/cuda_sifa/DEFCOR-AGG.egg-info
rm pytorchvideo/functional/pointwise_conv3d/*.so
rm -r pytorchvideo/functional/pointwise_conv3d/build

cd functional/cuda_sifa
python setup.py build develop
cd ../depthwise_conv3d
python3 setup.py build_ext --inplace
cd ../../
