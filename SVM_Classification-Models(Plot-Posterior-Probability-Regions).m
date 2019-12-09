%Plot Posterior Probability Regions for SVM Classification Models

%{
This example shows how to predict posterior probabilities of SVM models over a grid of observations, 
and then plot the posterior probabilities over the grid. Plotting posterior probabilities exposes decision boundaries.
%}

%Load Fisher's iris data set. Train the classifier using the petal lengths and widths, and remove the virginica species from the data.

load fisheriris
classKeep = ~strcmp(species,'virginica');
X = meas(classKeep,3:4);
y = species(classKeep);

%Train an SVM classifier using the data. It is good practice to specify the order of the classes.
SVMModel = fitcsvm(X,y,'ClassNames',{'setosa','versicolor'});

%Estimate the optimal score transformation function.
rng(1); % For reproducibility
[SVMModel,ScoreParameters] = fitPosterior(SVMModel); 

%Warning: Classes are perfectly separated. The optimal score-to-posterior transformation is a step function.
ScoreParameters

%{
The optimal score transformation function is the step function because the classes are separable. The fields LowerBound 
and UpperBound of ScoreParameters indicate the lower and upper end points of the interval of scores corresponding to observations 
within the class-separating hyperplanes (the margin). No training observation falls within the margin. If a new score is in the interval, 
then the software assigns the corresponding observation a positive class posterior probability, i.e., the value in the PositiveClassProbability field of ScoreParameters.

Define a grid of values in the observed predictor space. Predict the posterior probabilities for each instance in the grid.
%}
xMax = max(X);
xMin = min(X);
d = 0.01;
[x1Grid,x2Grid] = meshgrid(xMin(1):d:xMax(1),xMin(2):d:xMax(2));

[~,PosteriorRegion] = predict(SVMModel,[x1Grid(:),x2Grid(:)]);

%Plot the positive class posterior probability region and the training data
figure;
contourf(x1Grid,x2Grid,...
        reshape(PosteriorRegion(:,2),size(x1Grid,1),size(x1Grid,2)));
h = colorbar;
h.Label.String = 'P({\it{versicolor}})';
h.YLabel.FontSize = 16;
caxis([0 1]);
colormap jet;

hold on
gscatter(X(:,1),X(:,2),y,'mc','.x',[15,10]);
sv = X(SVMModel.IsSupportVector,:);
plot(sv(:,1),sv(:,2),'yo','MarkerSize',15,'LineWidth',2);
axis tight
hold off

%{
In two-class learning, if the classes are separable, then there are three regions: one where observations have positive class posterior probability 0, one where it is 1, 
and the other where it is the positive class prior probability
%}