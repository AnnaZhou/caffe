#!/bin/bash

set -x

export LD_LIBRARY_PATH=/mnt/data/workspace/OpenCV-2.4.2/lib:/opt/OpenBLAS/lib/

source /mnt/data/workspace/caffe/bin/vgg_spatial.sh Rafting
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh RockClimbingIndoor
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh RopeClimbing
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh Rowing
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh SalsaSpin
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh ShavingBeard
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh Shotput
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh SkateBoarding
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh Skiing
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh Skijet
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh SkyDiving
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh SoccerJuggling
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh SoccerPenalty
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh StillRings
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh SumoWrestling
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh Surfing
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh Swing
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh TableTennisShot
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh TaiChi
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh TennisSwing
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh ThrowDiscus
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh TrampolineJumping
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh Typing
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh UnevenBars
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh VolleyballSpiking
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh WalkingWithDog
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh WallPushups
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh WritingOnBoard
source /mnt/data/workspace/caffe/bin/vgg_spatial.sh YoYo
