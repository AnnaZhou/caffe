% function [ model_alpha ] = lapsvm_train_wildlife_cnn1(training_label_vector, training_instance_matrix,K)
%---get lapsvm score for wildlife dataset
% 
%%% workspace : load Wildlife/silhouette/index.txt;
%%% run loadCNNfeature.m
% setting default paths
setpaths

%VOCinit;
% % generating default options
fid1=fopen(sprintf('/Users/Anna/Documents/MatlabCode/VOCdevkit/results/VOC2007/Main/comp1_imagenet-cnn-vggfc8-lap-tval-AP-linear.txt'),'w');

gamA = 0.1;gamI = 0.1;

 options=make_options('gamma_I',gamI,'gamma_A',gamA,'Kernel', 'rbf','NN',5,'KernelParam', 0.35); %gamI=0.01 histint linear
 options.Verbose=1;
 
%     data.X = double( [cnnfeat5Train; cnnfeat5Val] ); 
%  
%     fprintf('Computing Gram matrix and Laplacian...\n\n');
%     tic;
%     data.K=calckernel(options,data.X, data.X);
%     data.L=laplacian(options,data.X);
%     toc;
% 
%     X1 =  double( cnnfeat5Test );      
%   
%       
%     K1=calckernel(options, data.X, X1 );
%     
%     save VOC2007_cnnfeat5_rbf cnnfeat5Train cnnfeat5Val cnnfeat5Test data K1 X1
    
%  testfname= '/Users/Anna/Documents/MatlabCode/VOCdevkit/VOC2007/ImageSets/Main/test.txt';
%  test_ids=textread(testfname,'%s');
%         
% for gamA = 0.1: 0.5: 1
%     for gamI = 0.1 : 0.3 : 1
%         options=make_options('gamma_I',gamI,'gamma_A',gamA,'Kernel', 'linear','NN',15,'KernelParam', 0.35); %gamI=0.01
%         options.Verbose=1;
%         %
%         % % options.UseBias =1;
%         %
%  
%         for clindex = 1 : 20
%             clear hogFeatureTrainLabel; clear data.Y;
%             ValLabel = zeros(size(cnnfeat5Val,1),1);
%             cls=VOCopts.classes{clindex};
%             filename = sprintf(VOCopts.clsimgsetpath,cls,'train');
%             [ids,hogFeatureTrainLabel]=textread(filename,'%s %d');
%            
%             data.Y = [hogFeatureTrainLabel; ValLabel];
%             
%             fprintf('Training data...\n');
%             % training the classifier
%             tic;
%             %         options.Cg=1; % PCG
%             %         options.MaxIter=1000; % upper bound
%             %         options.CgStopType=1; % 'stability' early stop
%             %         options.CgStopParam=0.015; % tolerance: 1.5%
%             %         options.CgStopIter=3; % check stability every 3 iterations
%             % fprintf('Training LapSVM in the primal with Newton''s method...\n');
%             %         classifier=svmp(options,data);
%             fprintf('Training LapSVM in the primal with svm''s method...\n');
%             
%             classifier= lapsvmp(options,data);
%        %          classifier=laprlsc(options,data);
%             
%             toc;
%             
%             
%             % test the classifier
%             fprintf('\nTesting data...\n');
%             % X--N*Dim training samples, x1--M*Dim input samples
%             tic;
%             clear score; clear dec_values;
%             
%             ndim = size(X1,1);
%             score = zeros(ndim,1);
%             for i = 1 : size(classifier.alpha,1)
%                 score(1:ndim,1) = score(1:ndim,1) + K1(:,i)*classifier.alpha(i,1) + classifier.b;
%             end
%             toc;
%             
%             dec_values(1:ndim,1) = score(1:ndim,1);
%             
% %             cls=VOCopts.classes{clindex};
%             fname=sprintf('/Users/Anna/Documents/MatlabCode/VOCdevkit/results/VOC2007/Main/%s%s%s','comp1-cnn-lap-',cls,'.txt');
%             fid=fopen(fname,'w');
%             for n=1: ndim
%                 fprintf(fid,'%s %f\n',test_ids{n},dec_values(n,1));
%             end
%             fclose(fid);
%             [recall,prec,ap]=VOCevalclsCnnLap(VOCopts,'comp1-cnn-lap-',cls,true);
%             %         x(clindex).name = char(classes1(posIndex,:));
%             % 		x(clindex).positive_n = ntest;
%             % 		x(clindex).likelihoods = dec_values1;
%             fprintf(fid1,'%s %f %f\n',cls, gamI, ap);
%         end
%     end
% end
% fclose(fid1);
%-----------------------------------------------------------------------------------------

% num_predictions_per_image = 1; 
% [scores,pred]=sort(dec_values,2,'descend');
% pred = pred(:,1:num_predictions_per_image);
% 
% [scores1,pred1]=sort(dec_values1,2,'descend');
% pred1 = pred1(:,1:num_predictions_per_image);
% % % write out predicted labels
% dlmwrite('demo.val.pred.txt',pred,'delimiter',' ');
% dlmwrite('demo.val.pred1.txt',pred1,'delimiter',' ');
% % height_distance = 14 - AWA_wnid_joint_height_distance_matrix;
% %evaluation
% error_flat=zeros(num_predictions_per_image,1);
% error_hie=zeros(num_predictions_per_image,1);
% 
% for i=1:num_predictions_per_image
%     error_flat(i) = eval_flat('demo.val.pred.txt','C:/Database/AwA-features-phog/Code/test_class10_400_groundtruth.txt', i);
% %     error_hie (i) = eval_flat('demo.val.pred1.txt','C:/Database/AwA-features-phog/Code/test_class10_trainVal1600_groundtruth.txt', i);
% %     error_hie (i) = eval_hie('demo.val.pred.txt','C:/Database/AwA-features-phog/Code/test_class10_150_groundtruth.txt',...
% %            height_distance, i);
% end
% 
% disp('# guesses  vs flat error');
% disp([(1:num_predictions_per_image)',error_flat]);
% disp('# guesses vs hierarchical error');
% disp([(1:num_predictions_per_image)',error_hie]);

%%%-----------------------------------------------------------------------------------------
%       ValLabel = zeros(size(cnnfeatVal1,1),1);
       X1 =  double( cnnfeatTest2 ); 
       subImgTrainInfo = cnnfeatTrainLabel2;
       
       clear mytrainlabel;clear mytrainlabelnum; clear dec_values; clear labelRe;
      
%        for clindex = 1 : 20
%             clear hogFeatureTrainLabel; 
%             
%             cls=VOCopts.classes{clindex};
% %             filename = sprintf(VOCopts.clsimgsetpath,cls,'trainval');
% %             [ids,hogFeatureTrainLabel]=textread(filename,'%s %d');
%            
%             %data.Y = [hogFeatureTrainLabel; ValLabel];
%             myi=1;
%             for lindex=1:size(hogFeatureTrainLabel,1)
%                 if hogFeatureTrainLabel(lindex,1) == 1
%                    mytrainlabel(clindex,myi)=lindex;
%                    myi=myi+1;
%                 end
%             end
%             mytrainlabelnum(clindex,1)=myi-1;
%        end 
       myIndexTotal=1;
       for clindex = 1 : 19
            clear posX;clear negX; clear posLabel;clear negLabel; clear data;
            posi=1;
            for myi = 1:size(subImgTrainInfo,1)
                if subImgTrainInfo(myi,1) == clindex
                   posX(posi,:) =cnnfeatTrain2(myi,:);
                   posLabel(posi,1)=1;
                   posi=posi+1;
                end
            end
            for cllindex = (1):19
                if clindex == cllindex
                    continue;
                end
                negi=1;
                for myi = 1:size(subImgTrainInfo,1)
                     if subImgTrainInfo(myi,1) == cllindex
                        negX(negi,:) =cnnfeatTrain2(myi,:);
                        negLabel(negi,1)=-1;
                        negi=negi+1;
                     end
                end
                
%                   data.X=[posX;negX;cnnfeatVal1];
%                   data.Y = [posLabel; negLabel; ValLabel];
                  data.X=[posX;negX ];
                  data.Y = [posLabel; negLabel ];
    
                data.Y=(data.Y);   
                data.K=calckernel(options,data.X, data.X);
                data.L=laplacian(options,data.X);

                K1=calckernel(options, data.X, X1 );
                
            fprintf('Training data %d ...\n',myIndexTotal);
            % training the classifier
            tic;
            % fprintf('Training LapSVM in the primal with Newton''s method...\n');
            %         classifier=svmp(options,data);
            fprintf('Training LapSVM in the primal with svm''s method...\n');
            
            classifier= lapsvmp(options,data);
   %              classifier=laprlsc(options,data);
            
            toc;
            
            
            % test the classifier
            fprintf('\nTesting data...\n');
            % X--N*Dim training samples, x1--M*Dim input samples
            tic;
            clear score; 
            
            ndim = size(X1,1);
            score = zeros(ndim,1);
            for i = 1 : size(classifier.alpha,1)
                score(1:ndim,1) = score(1:ndim,1) + K1(:,i)*classifier.alpha(i,1) + classifier.b;
            end
            toc;
            
            dec_values(1:ndim,myIndexTotal) = score(1:ndim,1);
         
            labelRe(myIndexTotal,1)=clindex;
            labelRe(myIndexTotal,2)=cllindex;
             
            myIndexTotal=myIndexTotal+1;
            end %cllindex
       end %clindex
   
   
%%--------------------------------------
numTest=size(X1,1);
myindex=1;
nbins=19;
myscore=zeros(size(dec_values,1),nbins);
for i=1:19
    for j=1:18
        myscore(:,i)=myscore(:,i)+dec_values(:,(i-1)*18+j);
    end
end
myscore1(:,:)=myscore(:,:)./18;
[scores,pred]=sort(myscore1,2,'descend');

num=size(cnnfeatTest2,1);
count = zeros(1,19);
for j = 1:19
    for i = 1:num
        if pred(i,j)==cnnfeatTestLabel2(i,1)
        count(1,j) = count(1,j)+1;
        end
    end
    if j>=2
       count(1,j)=count(1,j)+count(1,j-1);
    end
end
mAP=count./numTest;

% for i=1:size(myscore,1)
    
   % myscore1(i,:)=myscore(i,:)./norm(myscore(i,:));
% end


% myscore1=(myscore1+1)./2;
% k=5;
% imgindex=1;
% clear imgscore;
% imgscore=zeros(4952,20);
% cindex='000001';
% for i = 1:size(subImgTestInfo,1)
%     if cindex==(subImgTestInfoName(i,:))
%        imgscore(imgindex,:)=imgscore(imgindex,:)+myscore1(i,:).^k;    
%     else
%        cindex= subImgTestInfoName(i,:)
%        imgindex=imgindex+1;
%        imgscore(imgindex,:)=imgscore(imgindex,:)+myscore1(i,:).^k; 
%     end
% end

% num=size(cnnfeatTest2,1);
% myindex=1;
%  for thld = -0.35 : 0.01 : 0.25
%      mycls=zeros(num,19);
%      for clindex = 1 : 19
%          count=0;
%          for i=1:num
%              if myscore1(i,clindex)>=thld
%                  mycls(i,clindex)=clindex;
%                  if cnnfeatTestLabel2(i,1)==clindex
%                      count=count+1;
%                  end
%              else
%                  if cnnfeatTestLabel2(i,1)~=clindex
%                      count=count+1;
%                  end
%              end
%          end
%          apALL(myindex,clindex)=count;
%      end
%      
%      myindex = myindex + 1;
% end
% fclose(fid1);
% apAll1=apALL./numTest;   
% ap=max(apAll1,[],1); mAP=mean(ap);


   
