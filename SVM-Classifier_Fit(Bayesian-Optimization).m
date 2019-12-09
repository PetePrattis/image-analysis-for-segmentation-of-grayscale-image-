%Optimize an SVM Classifier Fit Using Bayesian Optimization
%{
This example shows how to optimize an SVM classification using the fitcsvm function and OptimizeHyperparameters name-value pair. 
The classification works on locations of points from a Gaussian mixture model. 
In The Elements of Statistical Learning, Hastie, Tibshirani, and Friedman (2009), page 17 describes the model. 
The model begins with generating 10 base points for a "green" class, distributed as 2-D independent normals with mean (1,0) 
and unit variance. It also generates 10 base points for a "red" class, distributed as 2-D independent normals with mean (0,1) 
and unit variance. For each class (green and red), generate 100 random points as follows:

-Choose a base point m of the appropriate color uniformly at random.

-Generate an independent random point with 2-D normal distribution with mean m and variance I/5, where I is the 2-by-2 identity matrix. 
In this example, use a variance I/50 to show the advantage of optimization more clearly.
%}

%Generate the Points and Classifier
%Generate the 10 base points for each class

rng default % For reproducibility
grnpop = mvnrnd([1,0],eye(2),10);
redpop = mvnrnd([0,1],eye(2),10);

%View the base points
plot(grnpop(:,1),grnpop(:,2),'go')
hold on
plot(redpop(:,1),redpop(:,2),'ro')
hold off

%Since some red base points are close to green base points, it can be difficult to classify the data points based on location alone.

%Generate the 100 data points of each class.
redpts = zeros(100,2);grnpts = redpts;
for i = 1:100
    grnpts(i,:) = mvnrnd(grnpop(randi(10),:),eye(2)*0.02);
    redpts(i,:) = mvnrnd(redpop(randi(10),:),eye(2)*0.02);
end

%View the data points.
figure
plot(grnpts(:,1),grnpts(:,2),'go')
hold on
plot(redpts(:,1),redpts(:,2),'ro')
hold off

%Prepare Data For Classification

%Put the data into one matrix, and make a vector grp that labels the class of each point.
cdata = [grnpts;redpts];
grp = ones(200,1);
% Green label 1, red label -1
grp(101:200) = -1;

%Prepare Cross-Validation
%Set up a partition for cross-validation. This step fixes the train and test sets that the optimization uses at each step.

c = cvpartition(200,'KFold',10);

%Optimize the Fit

%{
To find a good fit, meaning one with a low cross-validation loss, set options to use Bayesian optimization. 
Use the same cross-validation partition c in all optimizations.
%}

%For reproducibility, use the 'expected-improvement-plus' acquisition function.

opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
    'AcquisitionFunctionName','expected-improvement-plus');
svmmod = fitcsvm(cdata,grp,'KernelFunction','rbf',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)
	
%Find the loss of the optimized model.
lossnew = kfoldLoss(fitcsvm(cdata,grp,'CVPartition',c,'KernelFunction','rbf',...
    'BoxConstraint',svmmod.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint,...
    'KernelScale',svmmod.HyperparameterOptimizationResults.XAtMinObjective.KernelScale))
	
%This loss is the same as the loss reported in the optimization output under "Observed objective function value".

%Visualize the optimized classifier.
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(cdata(:,1)):d:max(cdata(:,1)),...
    min(cdata(:,2)):d:max(cdata(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(svmmod,xGrid);
figure;
h = nan(3,1); % Preallocation
h(1:2) = gscatter(cdata(:,1),cdata(:,2),grp,'rg','+*');
hold on
h(3) = plot(cdata(svmmod.IsSupportVector,1),...
    cdata(svmmod.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend(h,{'-1','+1','Support Vectors'},'Location','Southeast');
axis equal
hold off	