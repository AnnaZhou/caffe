#!/usr/bin/env sh
# Create the imagenet leveldb inputs
# N.B. set the path to the imagenet train + val data dirs

TOOLS=../../build/tools
DATA=/Users/Anna/workspace/dev_branch/caffe-dev/examples/WildLife/Images

echo "Creating leveldb..."

$TOOLS/convert_imageset / $DATA/train.txt imagenet_train_leveldb
$TOOLS/convert_imageset / $DATA/val.txt imagenet_val_leveldb 

#GLOG_logtostderr=1 $TOOLS/convert_imageset $DATA/ $DATA/train.txt imagenet_train_leveldb 1

#GLOG_logtostderr=1 $TOOLS/convert_imageset.bin \
#    $DATA/ \
#    $DATA/val.txt \
#    imagenet_val_leveldb 1

echo "Done."
