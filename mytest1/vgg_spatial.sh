#!/bin/bash

folder_name=$1

#cd ~/

#HOME_DIR=$(cd $(dirname "$0") && pwd)

HOME_DIR=/mnt/data

#SOURCE_DATA_DIR=${HOME_DIR}/ci_alg_rtb/UCF101/ucf101_flow_img_tvl1_gpu_ZIP/

WORK_DIR=${HOME_DIR}/workspace/caffe/workdir_t/

INPUT_DIR=${HOME_DIR}/workspace/caffe/workdir_t/${1}/

OUTPUT_DIR=${HOME_DIR}/workspace/caffe/workdir_t/output/${1}/

#CAFFE_DIR=${HOME_DIR}/workspace/caffe/

INPUT_FILE_LIST=${WORK_DIR}input_2strlist_spatial.txt

OUTPUT_FILE_LIST=${WORK_DIR}output_2strlist_spatial.txt

make_dir(){
        while read line
        do
                export dname=`echo $line | awk '{print $1}'`
                mkdir -p "$dname"
                #echo $dname
        done < $1
}
#mkdir -p ${WORK_DIR}

cd ${WORK_DIR}

aws s3 cp s3://annazhou.github.io/ucf/avi/${1}.tar.gz ${WORK_DIR}

tar -zxvf ${1}.tar.gz

cd ${1}

ls > ${WORK_DIR}tmp.txt

sed -e "s~^~${INPUT_DIR}~" ${WORK_DIR}tmp.txt > $INPUT_FILE_LIST
sed -i -e "s/.avi//g" $INPUT_FILE_LIST

sed -i -e "s/$/\//" ${WORK_DIR}tmp.txt

#sed -i -e "s/.avi//g" ${WORK_DIR}tmp.txt

#sed -e "s~^~${INPUT_DIR}~" ${WORK_DIR}tmp.txt > $INPUT_FILE_LIST

sed -i -e "s/.avi//g" ${WORK_DIR}tmp.txt

sed -e "s~^~${OUTPUT_DIR}~" ${WORK_DIR}tmp.txt > $OUTPUT_FILE_LIST

make_dir $OUTPUT_FILE_LIST

/mnt/data/workspace/caffe/build/examples/cpp_classification/classification_spatial2.bin /mnt/data/UCF101/cuhk_action_spatial_vgg_16_deploy.prototxt /mnt/data/UCF101/cuhk_action_spatial_vgg_16_split1.caffemodel /mnt/data/workspace/caffe/VGG_mean.binaryproto data/ilsvrc12/synset_words.txt //mnt/data/workspace/caffe/data/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01
