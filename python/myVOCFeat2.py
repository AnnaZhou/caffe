import numpy as np
import matplotlib.pyplot as plt
from pylab import *

import caffe

caffe_root = '../'  # this file is expected to be in {caffe_root}/examples

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
#MODEL_FILE = '../examples/VOC2007/pascal_deploy.prototxt'
#PRETRAINED = '../examples/VOC2007/pascal_train_val__iter_5000'
#MODEL_FILE = '/Users/Anna/workspace/dev_branch/caffe-dev/examples/imagenet/imagenet_deploy.prototxt'
#PRETRAINED = '../../../caffe/examples/imagenet/caffe_reference_imagenet_model'
MODEL_FILE = '/Users/Anna/workspace/dev_branch/caffe-dev/examples/VGG/vgg_CNN_M_deploy.prototxt'
PRETRAINED='/Users/Anna/workspace/dev_branch/caffe-dev/examples/VGG/vgg_CNN_M.caffemodel'

IMAGE_FILE = '../../../caffe/data/VOCdevkit/VOCtrainval_06-Nov-2007/JPEGImages/008750.jpg'

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
#                        mean=caffe_root + '../../caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy',
                       channel_swap=(2,1,0),
                       input_scale=255)

net.set_phase_test()
net.set_mode_cpu()

#FLISTPATH='/Users/Anna/workspace/caffe/data/VOCdevkit/VOCtrainval_06-Nov-2007/sublist/'
#FeatPATH='/Users/Anna/workspace/caffe/data/VOCdevkit/VOCtrainval_06-Nov-2007/vggsubfeat6/'
FLISTPATH='/Users/Anna/workspace/caffe/data/VOCdevkit/VOCtest_06-Nov-2007/sublist/'
FeatPATH='/Users/Anna/workspace/caffe/data/VOCdevkit/VOCtest_06-Nov-2007/vggsubfeat8/'

loop=1
for loop in range(1,14977):
 
    finName=FLISTPATH+'imlist_test_'+str(loop)+'.txt'
    print finName
    fin=open(finName,'r')
    IMAGE_FILE=fin.readline()
    #print IMAGE_FILE
    fin.close()

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
    feat = net.blobs['prob'].data[0]
    for i in range(0,1000):
        for j in range(0,1):
             for k in range(0,1):
                 print >> f1, feat[i][j][k]
                
    #feat.tofile(foutName,sep="",format="%s")
   
    f1.close()
    del input_image
    del feat
