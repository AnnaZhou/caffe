#define USE_OPENCV

#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <stdio.h> 
#include <math.h>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);

std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);
 
private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);
  std::vector<float> myPredict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);
  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  //CHECK(num_channels)<<num_channels;
  //CHECK(num_channels_ == 3 || num_channels_ == 1)
  //<< "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
//  SetMean(mean_file);

  /* Load labels. */
//  std::ifstream labels(label_file.c_str());
//  CHECK(labels) << "Unable to open labels file " << label_file;
//  string line;
//  while (std::getline(labels, line))
//    labels_.push_back(string(line));

//  shared_ptr<caffe::Blob<float> > fc6_layer = net_->blob_by_name("conv1_1");
//  std::cout <<fc6_layer->count()<<"\n";
//  float val;
//  FILE *fp;
//  fp=fopen("fc6.txt","w");
//  for(int i=0;i<1000;i++)//fc6_layer->count();i++)
//   { 
//     val=fc6_layer->cpu_data()[i];
//     fprintf(fp,"%d %f\t",i, val);
//   }  
//   fclose(fp);
// shared_ptr<caffe::Blob<float> > fc8_layer = net_->blob_by_name("fc8");
//  // Blob<float>* output_layer = net_->output_blobs()[0];
//  // CHECK_EQ(labels_.size(), output_layer->channels())
//  //   << "Number of labels is different from the output layer dimension.";
//  fp=fopen("fc8.txt","w");
//  for(int i=0;i<fc8_layer->count();i++)
//   { fprintf(fp,"%d ",i);
//     val=fc8_layer->cpu_data()[i];
//     fprintf(fp,"%f\t",val);
//   }
//   fclose(fp);


}


static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
 // input_layer->Reshape(1, num_channels_,
//                       input_geometry_.height, input_geometry_.width);
  input_layer->Reshape(50,20,224,224);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
 // WrapInputLayer(&input_channels);

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
//   for (int i = 0; i < input_layer->channels(); ++i)
//   {
//    cv::Mat channel(height, width, CV_32FC1, input_data);
//    input_channels->push_back(channel);
//    input_data += width * height;
//   }

  const float img_to_net_scale = 0.0039215684;
//  TransformationParameter input_xform_param;
//  input_xform_param.set_scale( img_to_net_scale );
//  DataTransformer<float> input_xformer( input_xform_param, TEST );
//  input_xformer.Transform( img, &input_blob );
      std::vector<Blob<float>*> input;
//      input.push_back( &input_blob );

 // Preprocess(img, &input_channels);

  net_->Forward();

  shared_ptr<caffe::Blob<float> > fc6_layer = net_->blob_by_name("conv1_1");
  std::cout <<fc6_layer->count()<<"\n";
  float val;
  FILE *fp;
  fp=fopen("fc6.txt","w");
  for(int i=0;i<1000;i++)//fc6_layer->count();i++)
   { 
     val=fc6_layer->cpu_data()[i];
     fprintf(fp,"%d %f\t",i, val);
   }  
   fclose(fp);
     shared_ptr<caffe::Blob<float> > fc8_layer = net_->blob_by_name("fc8");
//  // Blob<float>* output_layer = net_->output_blobs()[0];
//  // CHECK_EQ(labels_.size(), output_layer->channels())
//  //   << "Number of labels is different from the output layer dimension.";
  fp=fopen("fc8.txt","w");
  for(int i=0;i<fc8_layer->count();i++)
   { fprintf(fp,"%d ",i);
     val=fc8_layer->cpu_data()[i];
     fprintf(fp,"%f\t",val);
   }
   fclose(fp);

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
 


  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
   for (int i = 0; i < input_layer->channels(); ++i)
   {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
   }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
//  cv::Mat sample;
//  if (img.channels() == 3 && num_channels_ == 1)
//    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
//  else if (img.channels() == 4 && num_channels_ == 1)
//    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
//  else if (img.channels() == 4 && num_channels_ == 3)
//    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
//  else if (img.channels() == 1 && num_channels_ == 3)
//    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
//  else
//    sample = img;

//  cv::Mat sample_resized;
//  if (sample.size() != input_geometry_)
//    cv::resize(sample, sample_resized, input_geometry_);
//  else
//    sample_resized = sample;

//  cv::Mat sample_float;
//  if (num_channels_ == 3)
//    sample_resized.convertTo(sample_float, CV_32FC3);
//  else
//    sample_resized.convertTo(sample_float, CV_32FC1);

//  cv::Mat sample_normalized;
//  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
//  cv::split(sample_normalized, *input_channels);

  cv::split(img, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";

}

int main(int argc, char** argv) {

std::cout<<"beging code:";
//caffe.set_mode_gpu();

if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);
std::cout<<"init";

  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file   = argv[4];
  Classifier classifier(model_file, trained_file, mean_file, label_file);
  
  std::cout<<"\n classifier code:";
  
  string file = argv[5];

  std::cout << "---------- Prediction for "
           << file << " ----------" ;//<< std::endl;
 //std::cout <<"predict";
 // cv::Mat img1 = cv::imread(file, -1);
 // CHECK(!img1.empty()) << "Unable to decode image " << file;
 // std::cout<<"my Code:"; 
  int i; 
  int num_frame=40; //atoi(argv[6]);
  if(num_frame==0){num_frame=40;} 
  string line;
 // FILE *inlist;
  // ifstream myfile(file);
//  DIR *dir;
//  struct dirent *ent;

    int imgSize[]={256,340};
    cv::Mat myimg(256,340,CV_32FC1);
    int histSize[] = {25, 20,256,340 };
   // cv::Mat flow_x(4, histSize, CV_32FC1, cv::Scalar(255));
   // cv::Mat flow_f(4, histSize, CV_32FC1, cv::Scalar(255));
    histSize[0]=25; histSize[2] =224;histSize[3]=224; 
    int hSize[] = {50, 20, 224, 224 };
    cv::Mat img(4, hSize, CV_32F);//, cv::Scalar(255));
   // cv::Mat img(25,20,224,224,CV_32F);    
   std::cout<<"init end";
    //inlist=fopen("inputlist_temporal.txt","r");
     int j=0;
     int k;int l;
     int m;int n;
     char myfile[300];
     int fidx; int fldx;
     int idx[4];int idex[3];
    //  myfile="/mnt/data/UCF101/flow";
     num_frame=40;
    // int step = 4;
     int step=int(((num_frame-10+1)/25));
    FILE* infl;
    infl=fopen("myinput.txt","w");
    std::cout<< 1 ;
    cv::Mat myimg1;
    IplImage* myimg2 = 0; 
    uchar *data; int channels; int istep;
   // if(inlist!=NULL)
   // {
   //  while( feof(inlist) )
   //   {
       // std::cout<< fidx; 
       for(fidx=0;fidx<25;fidx++)
        {
         for(fldx=0;fldx<10;fldx++)
         {
           std::cout<< fidx;
           l=fidx*step+fldx;
                      
          // sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l);
          // cv::Mat myimg0 = cv::imread(myfile,-1); 
           sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+1);
         //  std::cout<< "imread:";
         //   myimg1 = cvLoadImageM(myfile,0);
            idex[0]=0;   idex[1]=0;  idex[0]=0; 
            
            myimg2=cvLoadImage(myfile,0);
            data=(uchar*)myimg2->imageData;
            channels= myimg2->nChannels;
            istep=myimg2->widthStep;
         // std::cout<<"img op:" ;        
           for(i=0;i<224;i++)
            for(j=0;j<224;j++)
             {
               idx[0]=fidx;idx[1]=2*fldx;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
               img.at<float>(idx) = data[i*istep+j*channels];  // myimg1.at<float>(idex);
               k = data[i*istep+j*channels]; //   std::cout<<data[i*istep+j*channels];  // std::cout<<i,j,img.at<float>(idx) ;
              }
            std::cout<<"data"<<(k);//<<(data[0*istep+0*channels]);
            cvReleaseImage(&myimg2);
            sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+1);
            myimg2 = cvLoadImage(myfile,0);        
           for(i=0;i<224;i++)
            for(j=0;j<224;j++)
             {
               idx[0]=fidx;idx[1]=2*fldx+1;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
              // img.at<float>(idx) = myimg1.at<float>(idex);
               img.at<float>(idx) = data[i*istep+j*channels]; 
              }           
              cvReleaseImage(&myimg2);
           }//for fldx;
          }//for fidx;
         // std::cout<<idx[0]<<idx[1]<<idx[2];
         // fclose(infl);
        // cv::Mat myimg1 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+2);
        //   cv::Mat myimg2 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+3);
        //   cv::Mat myimg3 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+4);
        //   cv::Mat myimg4 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+5);
        //   cv::Mat myimg5 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+6);
        //   cv::Mat myimg6 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+7);
        //   cv::Mat myimg7 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+8);
        //   cv::Mat myimg8 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+9);
        //   cv::Mat myimg9 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+10);
        //   cv::Mat myimg10 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+11);
        //   cv::Mat myimg11 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+12);
        //   cv::Mat myimg12 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+13);
        //   cv::Mat myimg13 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+14);
        //   cv::Mat myimg14 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+15);
        //   cv::Mat myimg15 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+16);
        //   cv::Mat myimg16 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+17);
        //   cv::Mat myimg17 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+18);
        //   cv::Mat myimg18 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+19);
        //   cv::Mat myimg19 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+20);
        //   cv::Mat myimg20 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+21);
        //   cv::Mat myimg21 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+22);
        //   cv::Mat myimg22 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+23);
        //   cv::Mat myimg23 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+24);
        //   cv::Mat myimg24 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_x_%04d.jpg",l+25);
        //   cv::Mat myimg25 = cv::imread(myfile,-1);
        //   sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+1);
        //   cv::Mat myimg26 = cv::imread(myfile,-1);
       //    sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+2);
       //    cv::Mat myimg27 = cv::imread(myfile,-1);
       //    sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+3);
       //    cv::Mat myimg28 = cv::imread(myfile,-1);
       //    sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+4);
       //    cv::Mat myimg29 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+5);
      //     cv::Mat myimg30 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+6);
      //     cv::Mat myimg31 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+7);
      //     cv::Mat myimg32 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+8);
      //     cv::Mat myimg33 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+9);
      //     cv::Mat myimg34 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+10);
      //     cv::Mat myimg35 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+11);
      //     cv::Mat myimg36 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+12);
      //     cv::Mat myimg37 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+13);
      //     cv::Mat myimg38 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+14);
      //     cv::Mat myimg39 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+15);
      //     cv::Mat myimg40 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+16);
      //     cv::Mat myimg41 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+17);
      //     cv::Mat myimg42 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+18);
      //     cv::Mat myimg43 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+19);
      //     cv::Mat myimg44 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+20);
      //     cv::Mat myimg45 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+21);
      //     cv::Mat myimg46 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+22);
      //     cv::Mat myimg47 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+23);
      //     cv::Mat myimg48 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+24);
      //     cv::Mat myimg49 = cv::imread(myfile,-1);
      //     sprintf(myfile,"/mnt/data/UCF101/flow/YoYo/v_YoYo_g01_c01/flow_y_%04d.jpg",l+25);
      //     cv::Mat myimg50 = cv::imread(myfile,-1);

        // int idx[4];int idex[2]; 
        //  for(l=0;l<20;l++)
        //   for(i=0;i<224;i++)
        //    for(j=0;j<224;j++)
        //     {
        //       idx[0]=0;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg1.at<float>(idex);
        //       idx[0]=2;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg2.at<float>(idex);
        //       idx[0]=4;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg3.at<float>(idex);
        //       idx[0]=6;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg4.at<float>(idex);               
        //       idx[0]=8;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg5.at<float>(idex);
        //       idx[0]=10;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg6.at<float>(idex);
        //       idx[0]=12;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg7.at<float>(idex);
        //       idx[0]=14;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg8.at<float>(idex);
        //       idx[0]=16;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg9.at<float>(idex);
        //       idx[0]=18;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg10.at<float>(idex);
        //       idx[0]=20;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg11.at<float>(idex);
        //       idx[0]=22;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg12.at<float>(idex);
        //       idx[0]=24;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg13.at<float>(idex);
        //       idx[0]=26;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg14.at<float>(idex);
        //       idx[0]=28;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg15.at<float>(idex);
        //       idx[0]=30;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg16.at<float>(idex);
        //       idx[0]=32;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg17.at<float>(idex);
        //       idx[0]=34;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg18.at<float>(idex);
        //       idx[0]=36;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg19.at<float>(idex);
        //       idx[0]=38;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg20.at<float>(idex);
        //       idx[0]=40;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg21.at<float>(idex);
        //       idx[0]=42;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg22.at<float>(idex);
        //       idx[0]=44;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg23.at<float>(idex);
        //       idx[0]=46;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg24.at<float>(idex);
        //       idx[0]=48;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg25.at<float>(idex);              
        //       idx[0]=1;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg1.at<float>(idex);
        //       idx[0]=3;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg2.at<float>(idex);
        //       idx[0]=5;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg3.at<float>(idex);
        //       idx[0]=7;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg4.at<float>(idex);
        //       idx[0]=9;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg5.at<float>(idex);
        //       idx[0]=11;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg6.at<float>(idex);
        //       idx[0]=13;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg7.at<float>(idex);
        //       idx[0]=15;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg8.at<float>(idex);
        //       idx[0]=17;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg9.at<float>(idex);
        //       idx[0]=19;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg10.at<float>(idex);
        //       idx[0]=21;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg11.at<float>(idex);
        //       idx[0]=23;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg12.at<float>(idex);
        //       idx[0]=25;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg13.at<float>(idex);
        //       idx[0]=27;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg14.at<float>(idex);
        //       idx[0]=29;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg15.at<float>(idex);
        //       idx[0]=31;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg16.at<float>(idex);
        //       idx[0]=33;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg17.at<float>(idex);
        //       idx[0]=35;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg18.at<float>(idex);
        //       idx[0]=37;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg19.at<float>(idex);
        //       idx[0]=39;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg20.at<float>(idex);
        //       idx[0]=41;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg21.at<float>(idex);
        //       idx[0]=43;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg22.at<float>(idex);
        //       idx[0]=45;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg23.at<float>(idex);
        //       idx[0]=47;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg24.at<float>(idex);
        //       idx[0]=49;idx[1]=l;idx[2]=i;idx[3]=j;idex[0]=i; idex[1]=j;
        //       img.at<float>(idx) = myimg25.at<float>(idex);      
         
        //      }

        //  }
        // }    
           
     //     }
    // }


  std::vector<Prediction> predictions = classifier.Classify(img); 



  /* Print the top N predictions. */
//  for (size_t i = 0; i < predictions.size(); ++i) {
//    Prediction p = predictions[i];
//    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
//              << p.first << "\"" << std::endl;
//  }

//img1.release();
myimg.release();
//flow_x.release();
//flow_f.release();
img.release();
//myimg.release();


}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
