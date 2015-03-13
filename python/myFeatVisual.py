import numpy as np
import matplotlib.pyplot as plt
from pylab import *

import caffe

caffe_root = '../'  # this file is expected to be in {caffe_root}/examples

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
#MODEL_FILE = '../examples/VOC2007/pascal_deploy.prototxt'
#PRETRAINED = '../examples/VOC2007/pascal_train_val__iter_5000'

#MODEL_FILE = '/Users/Anna/workspace/caffe/examples/imagenet/imagenet_deploy.prototxt'
#PRETRAINED = '/Users/Anna/workspace/caffe/examples/imagenet/caffe_reference_imagenet_model'

MODEL_FILE = '/Users/Anna/workspace/dev_branch/caffe-dev/examples/VGG/vgg_CNN_M_deploy.prototxt'
#PRETRAINED='/Users/Anna/workspace/dev_branch/caffe-dev/examples/WildLife/caffe_wildlife_nev_train_iter_10000.caffemodel'
PRETRAINED='/Users/Anna/workspace/dev_branch/caffe-dev/examples/VGG/VGG_CNN_M.caffemodel';

#IMAGE_FILE = '/Users/Anna/workspace/dev_branch/caffe-dev/examples/images/cat.jpg'; 
#IMAGE_FILE = '../data/VOCdevkit/VOCtrainval_06-Nov-2007/JPEGImages/008750.jpg'
IMAGE_FILE = '/Users/Anna/Documents/MatlabCode/Wildlife/002434.jpg'
#IMAGE_FILE = '/Users/Anna/Documents/MatlabCode/Wildlife/000018.jpg'
#IMAGE_FILE = '/Users/Anna/Documents/MatlabCode/Wildlife/206.jpg'

net = caffe.Classifier(MODEL_FILE, PRETRAINED)
#                        #mean_file=caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy',
#                       channel_swap=(2,1,0),
#                       input_scale=255)

net.set_phase_test()
net.set_mode_cpu()
net.set_mean('data',np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))
net.set_raw_scale('data',255)
net.set_channel_swap('data',(2,1,0))

#FLISTPATH='/Users/Anna/workspace/caffe/data/VOCdevkit/VOCtrainval_06-Nov-2007/sublist/'
#FeatPATH='/Users/Anna/workspace/caffe/data/VOCdevkit/VOCtrainval_06-Nov-2007/subfeat6/'
#FLISTPATH='/Users/Anna/workspace/caffe/data/VOCdevkit/VOCtest_06-Nov-2007/sublist/'
#FeatPATH='/Users/Anna/workspace/caffe/data/VOCdevkit/VOCtest_06-Nov-2007/subfeat6/'
FLISTPATH='/Users/Anna/Documents/MatlabCode/Wildlife/silhouette/list/'
FeatPATH='/Users/Anna/Documents/MatlabCode/Wildlife/silhouette/fc6sgd/'

def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)

loop=1
for loop in range(1,2):
 
    finName=FLISTPATH+'imlist_'+str(loop)+'.txt'
#    print finName
#    fin=open(finName,'r')
#    IMAGE_FILE=fin.readline()
    #print IMAGE_FILE
#    fin.close()
    print IMAGE_FILE 
    input_image = caffe.io.load_image(IMAGE_FILE)
    #plt.imshow(input_image)
    #plt.show()

    #print 'predict...'
    prediction = net.predict([input_image]) 

    #print(prediction)
    #print 'prediction shape:', prediction[0].shape
    #plt.plot(prediction[0])
    #plt.show()

    foutName = FeatPATH+str(loop)+'.txt'
    f1=open(foutName, 'w')

    #[(k,v.data.shape) for k, v in net.blobs.items()]
    #print >> f1,k
    #print >> f1,v
    feat = net.blobs['fc6'].data[0]
#    for i in range(0,4092):
#        for j in range(0,1):
#             for k in range(0,1):
#                 print >> f1, feat[i][j][k]
#    filters = net.blobs['conv3'].data[4]
#    vis_square(feat,padval=0.5)
#    filters = net.blobs['conv1'].data[4, :36]
    
    filters = net.params['conv1'][0].data
    vis_square(filters.transpose(0,2,3,1))
#    feat = net.blobs['conv4'].data[4]    
#    vis_square(feat, padval=0.5)
    plt.show()
 
#    feat = net.blobs['fc7'].data[4]
#    plt.subplot(2, 1, 1)
#    plt.plot(feat.flat)
#    plt.subplot(2, 1, 2)
#    _ = plt.hist(feat.flat[feat.flat > 0], bins=100)
#    plt.show()
    #feat.tofile(foutName,sep="",format="%s")
   
    f1.close()
    del input_image
    del feat
