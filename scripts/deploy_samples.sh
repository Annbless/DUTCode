#/bin/bash

OutputBasePath='results/'
SmootherPath='ckpt/smoother.pth'
RFDetPath='ckpt/RFDet_640.pth.tar'
PWCNetPath='ckpt/network-default.pytorch'
MotionProPath='ckpt/MotionPro.pth'
DIFPath='ckpt/DIFNet2.pth'
StabNetPath='ckpt/stabNet.pth'
InputPath='images/'


if [ -d "$OutputBasePath" ]; then
    echo "Directory exists" ;
else
    `mkdir -p $OutputBasePath`;
fi

# Run the DUT model
echo " Stabiling using the DUT model "
echo "-----------------------------------"

python ./scripts/DUTStabilizer.py \
	--SmootherPath=$SmootherPath \
    --RFDetPath=$RFDetPath \
    --PWCNetPath=$PWCNetPath \
    --MotionPro=$MotionProPath \
    --InputBasePath=$InputPath \
    --OutputBasePath=$OutputBasePath 

# Run the DIFRINT model
echo " Stabiling using the DIFRINT model "
echo "-----------------------------------"

python ./scripts/DIFRINTStabilizer.py \
    --modelPath=$DIFPath \
    --InputBasePath=$InputPath \
    --OutputBasePath=$OutputBasePath \
    --cuda 

# Run the StabNet model
echo " Stabiling using the DIFRINT model "
echo "-----------------------------------"

python ./scripts/StabNetStabilizer.py \
    --modelPath=$StabNetPath \
    --OutputBasePath=$OutputBasePath \
    --InputBasePath=$InputPath 

