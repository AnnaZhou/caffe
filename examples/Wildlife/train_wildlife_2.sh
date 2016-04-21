#! /bin/bash
set -x

TOOLS=../../build/tools
$TOOLS/caffe train --solver='imagenet_solver.prototxt' --model='imagenet_train.prototxt'
#GLOG_logtostderr=1 $TOOLS/caffe train --solver='imagenet_solver.prototxt' --model='imagenet_train.prototxt'
echo "Done."
