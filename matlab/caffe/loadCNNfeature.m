%%%---load /animal/silhouette/index.txt

myfName = char(VarName1(:,:));
myfNum  = VarName2(:,:);

imgpath = '/Users/Anna/Documents/MatlabCode/Wildlife/silhouette';
%datapath = '/Users/Anna/Documents/MatlabCode/Wildlife/silhouette/list';
datapath='/Users/Anna/workspace/dev_branch/caffe-dev/examples/WildLife/Images';
featpath= '/Users/Anna/Documents/MatlabCode/Wildlife/silhouette/vggfc8';

fNum=1;
% fname= sprintf('%s/train.txt',datapath);
% fileA=fopen(fname,'w');
%    
% fname= sprintf('%s/val.txt',datapath);
% fileB=fopen(fname,'w');

Wid=256;

 for i = 1 : 19
     for ii=1 : myfNum(i,1)
         imIndex=200+ii;
%         imName= sprintf('%s/%s/%d_gauss.jpg',imgpath,char(VarName1{i}),imIndex);
%         img=imread(imName);
% %         imshow(img);
%         if size(img,3) == 3
%             subImg1(:,:,1)=imresize(img(:,:,1), [Wid Wid]);
%             subImg1(:,:,2)=imresize(img(:,:,2), [Wid Wid]);
%             subImg1(:,:,3)=imresize(img(:,:,3), [Wid Wid]);
%         else
%             subImg1(:,:,1)=imresize(img, [Wid Wid]);
%             subImg1(:,:,2)=subImg1(:,:,1);
%             subImg1(:,:,3)=subImg1(:,:,1);
%         end
%         imSubName = sprintf('%s/%d.jpg',datapath,fNum );
%         imwrite( subImg1,imSubName);
%         
%         if mod(ii,2) == 0
%         fprintf(fileA,'%s %d\n',imSubName,(i-1));
%         else
%          fprintf(fileB,'%s %d\n',imSubName,(i-1));   
%         end
        
        wildlife(fNum,1)=i; wildlife(fNum,2)=ii;
         fNum=fNum+1;
     end
 end
%        fclose(fileA);




clear feat; clear cnnfeat; clear cnnfeatTrain2;clear cnnfeatTest2;
%%generate training and testing features
num=691;
for i = 1 : num
    filename = sprintf('%s/%d.txt',featpath,i);
    feat1=textread(filename,'%f');
    feat(i,:)=feat1';
end
for i = 1:size(feat,1)
    feat1=feat(i,:);
    cnnfeat(i,:)=feat1./norm(feat1);
end
trainIndex=1; testIndex=1;
for i=1:num
    if mod(wildlife(i,2),2)==1
       cnnfeatTrain2(trainIndex,:)=cnnfeat(i,:);
       cnnfeatTrainLabel2(trainIndex,1)=wildlife(i,1);
       trainIndex=trainIndex+1;
       
    else
       cnnfeatTest2(testIndex,:)=cnnfeat(i,:);
       cnnfeatTestLabel2(testIndex,1)=wildlife(i,1);
       testIndex =testIndex+1;
    end
end
