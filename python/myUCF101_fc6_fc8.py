import os.path
import numpy as np
import caffe
import glob
import os
import numpy as np
import math
import cv2
import scipy.io as sio
import time

MODEL_FILE ='/mnt/data/workspace/caffe/bin/cuhk_action_temporal_vgg_16_flow_deploy.prototxt'
PRETRAINED ='/mnt/data/workspace/caffe/bin/cuhk_action_temporal_vgg_16_split3.caffemodel'

net = caffe.Net(MODEL_FILE,PRETRAINED,caffe.TEST)

finNamex='/mnt/data/workspace/caffe/workdir_t/'+'input_2strlist_flow.txt'
foutName='/mnt/data/workspace/caffe/workdir_t/'+'output_2strlist_flow.txt'

fout=open(foutName,'r')
finx=open(finNamex,'r')

IMAGE_FILE=finx.readline()
while IMAGE_FILE:
    img_fd=IMAGE_FILE.rstrip('\n')
    img_fd=img_fd.rstrip('\r')
    print img_fd
    num_samples=25
    num_frames=0
    optical_flow_frames=10
    start_frame=0

    if num_frames == 0:
       imglist = glob.glob(os.path.join(img_fd, '*flow_x*.jpg'))
       duration = len(imglist)
    else:
       duration = num_frames
    step = int(math.floor((duration-optical_flow_frames+1)/num_samples))  

    dims = (256,340,optical_flow_frames*2,num_samples)
    flow = np.zeros(shape=dims, dtype=np.float64)
    flow_flip = np.zeros(shape=dims, dtype=np.float64)

    print(time.time(), time.clock())
    for i in range(num_samples):
        for j in range(optical_flow_frames):
            flow_x_file = os.path.join(img_fd, 'flow_x_{0:04d}.jpg'.format(i*step+j+1 + start_frame))
            flow_y_file = os.path.join(img_fd, 'flow_y_{0:04d}.jpg'.format(i*step+j+1 + start_frame))
            #img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
            #img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
            #img_x = cv2.resize(img_x, dims[1::-1])
            #img_y = cv2.resize(img_y, dims[1::-1])
            img_x=caffe.io.load_image(flow_x_file,color=False)
            img_y=caffe.io.load_image(flow_y_file,color=False)
            img_x=img_x[:,:,0]*255
            img_y=img_y[:,:,0]*255
            flow[:,:,j*2  ,i] = img_x[:,:]
            flow[:,:,j*2+1,i] = img_y[:,:]
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
    
    
    print(time.time(), time.clock())

    batch_size = 50
    num_batches = int(math.ceil(float(flow.shape[3])/batch_size))

    feat_FILE=fout.readline()
    feat_fn=feat_FILE.rstrip('\n')
    feat_fn=feat_fn.rstrip('\r') 
    feat6=np.zeros(4096,dtype=np.float32)
    feat8=np.zeros(101,dtype=np.float32) 
    for bb in range(num_batches): 
        span = range(batch_size*bb, min(flow.shape[3],batch_size*(bb+1)))
        net.blobs['data'].data[...] = np.transpose(flow[:,:,:,span], (3,2,1,0))
        output = net.forward()
        feat = net.blobs['fc8'].data[0]
        feat8=feat8+feat
        feat_f=feat_fn+'fc8_%d.txt' % (bb+1)
        ffile=open(feat_f, 'w')
        #print >>ffile, feat
        for i in range(0,101):
            print >> ffile, feat[i]
        ffile.close()
        feat = net.blobs['fc6'].data[0]
        feat6=feat6+feat
        feat_f=feat_fn+'fc6_%d.txt' % (bb+1)
        ffile=open(feat_f, 'w')
        #print >>ffile, feat
        for i in range(0,4096):
            print >> ffile, feat[i]
        ffile.close()
    feat_f=feat_fn+'fc8_a.txt'
    ffile=open(feat_f, 'w')
    for i in range(0,101):
        print >> ffile, feat8[i]
    ffile.close()
    feat_f=feat_fn+'fc6_a.txt'
    ffile=open(feat_f, 'w')
    for i in range(0,4096):
        print >> ffile, feat6[i]
    ffile.close()
    IMAGE_FILE=finx.readline()
print(time.time(), time.clock())

finx.close()
fout.close()
