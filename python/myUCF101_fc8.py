import os.path
import numpy as np

import caffe
MODEL_FILE ='/mnt2/data/ci_alg_rtb/UCF101/src/python/cuhk_action_temporal_vgg_16_flow_deploy.prototxt'
PRETRAINED ='/mnt2/data/ci_alg_rtb/UCF101/cuhk_action_temporal_vgg_16_split1.caffemodel'
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
#                       mean='/Users/Anna/workspace/dev_branch/caffe-dev/examples/twoStreams/imagenet_mean.binaryproto', 
#                        mean=caffe_root + '../../caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy',
#                       channel_swap=(3,2,1,0),
                       input_scale=255)
                       
finNamex='/mnt2/data/ci_alg_rtb/UCF101/src/python/'+'input_2strlist_flow1.txt'
finx=open(finNamex,'r')
foutName='/mnt2/data/ci_alg_rtb/UCF101/src/python/'+'output_2strlist_flow1.txt'

num_samples=25
num_frames=0
optical_flow_frames=10
start_frame=0
import glob
import os
import numpy as np
import math
import cv2
import scipy.io as sio


IMAGE_FILE=finx.readline()
img_fd=IMAGE_FILE.rstrip('\n')
print img_fd
if num_frames == 0:
    imglist = glob.glob(os.path.join(img_fd, '*flow_x*.jpg'))
    duration = len(imglist)
else:
    duration = num_frames
step = int(math.floor((duration-optical_flow_frames+1)/num_samples))  

#input_img=np.zeros((25,20,256,340))
#in_img=np.zeros((250,20,224,224))

dims = (256,340,optical_flow_frames*2,num_samples)

flow = np.zeros(shape=dims, dtype=np.float64)
flow_flip = np.zeros(shape=dims, dtype=np.float64)

for i in range(num_samples):
    for j in range(optical_flow_frames):
        flow_x_file = os.path.join(img_fd, 'flow_x_{0:04d}.jpg'.format(i*step+j+1 + start_frame))
        flow_y_file = os.path.join(img_fd, 'flow_y_{0:04d}.jpg'.format(i*step+j+1 + start_frame))
        img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
        img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
        img_x = cv2.resize(img_x, dims[1::-1])
        img_y = cv2.resize(img_y, dims[1::-1])
        flow[:,:,j*2  ,i] = img_x
        flow[:,:,j*2+1,i] = img_y
        flow_flip[:,:,j*2  ,i] = 255 - img_x[:, ::-1]
        flow_flip[:,:,j*2+1,i] = img_y[:, ::-1]

# crop, Negative numbers mean that you count from the right instead of the left. 
#       So, list[-1] refers to the last element, list[-2] is the second-last, and so on.
flow_1 = flow[:224, :224, :,:]
flow_2 = flow[:224, -224:, :,:]
flow_3 = flow[16:240, 60:284, :,:]
flow_4 = flow[-224:, :224, :,:]
flow_5 = flow[-224:, -224:, :,:]
flow_f_1 = flow_flip[:224, :224, :,:]
flow_f_2 = flow_flip[:224, -224:, :,:]
flow_f_3 = flow_flip[16:240, 60:284, :,:]
flow_f_4 = flow_flip[-224:, :224, :,:]
flow_f_5 = flow_flip[-224:, -224:, :,:]
del flow
flow = np.concatenate((flow_1,flow_2,flow_3,flow_4,flow_5,flow_f_1,flow_f_2,flow_f_3,flow_f_4,flow_f_5), axis=3)
flow=flow-128
flow = np.transpose(flow, (1,0,2,3))
    
batch_size = 50
num_batches = int(math.ceil(float(flow.shape[3])/batch_size))
#fout.close()
fout=open(foutName,'r')
feat_FILE=fout.readline()
feat_fn=feat_FILE.rstrip('\n')
featfc8=np.zeros(101)    
for bb in range(num_batches):
    span = range(batch_size*bb, min(flow.shape[3],batch_size*(bb+1)))
    net.blobs['data'].data[...] = np.transpose(flow[:,:,:,span], (3,2,1,0))
    output = net.forward()

    feat = net.blobs['fc8'].data[0]
    featfc8=featfc8+feat

feat_f=feat_fn+'_%d.txt' % (11)
ffile=open(feat_f, 'w')
print feat_f
for i in range(0,101):
    print >> ffile, featfc8[i]
ffile.close()

finx.close()
fout.close()
