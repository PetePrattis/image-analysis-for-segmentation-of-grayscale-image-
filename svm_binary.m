X = 1:length(cluster_idx);
X = X(:);
X = [X cluster_idx];

X= ab;
Y = cluster_idx;
X = X(2:100001,:);
Y = Y(1:100000,:);



load carsmall
rng 'default'  % For reproducibility
%Specify Horsepower and Weight as the predictor variables (X) and MPG as the response variable (Y).
X = [Horsepower Weight];
Y = MPG;
%Cross-validate two SVM regression models using 5-fold cross-validation. For both models, specify to standardize the predictors. 
%For one of the models, specify to train using the default linear kernel, and the Gaussian kernel for the other model.
MdlLin = fitrsvm(X,Y,'Standardize',true,'KFold',5)
MdlGau = fitrsvm(X,Y,'Standardize',true,'KFold',5,'KernelFunction','gaussian')
MdlLin.Trained
%Compare the generalization error of the models. In this case, the generalization error is the out-of-sample mean-squared error.
mseLin = kfoldLoss(MdlLin)
mseGau = kfoldLoss(MdlGau)
%The SVM regression model using the Gaussian kernel performs better than the one using the linear kernel.
%Create a model suitable for making predictions by passing the entire data set to fitrsvm, 
%and specify all name-value pair arguments that yielded the better-performing model. However, do not specify any cross-validation options.
MdlGau = fitrsvm(X,Y,'Standardize',true,'KernelFunction','gaussian');
rng default
%Find hyperparameters that minimize five-fold cross-validation loss by using automatic hyperparameter optimization.
%For reproducibility, set the random seed and use the 'expected-improvement-plus' acquisition function.
Mdl = fitrsvm(X,Y,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'))