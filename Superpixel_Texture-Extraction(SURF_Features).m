%acquire image
source = imread("vegetables.jpg");

I = rgb2gray(source);
%Ilab = rgb2lab(source);

[L,N] = superpixels(source,100);

BW = boundarymask(L);

outputImage = zeros(size(source), 'like', source);

%{
I'll use the function label2idx to compute the indices of the pixels in each superpixel cluster. 
That will let me access the red, green, and blue component values using linear indexing
%}
idx = label2idx(L);
numRows = size(source,1);
numCols = size(source,2);

%{
For each of the N superpixel clusters, use linear indexing to access the red, green, and blue components, 
reconstruct the corresponding pixels while detecting SURF Features for the grayscale version of the image
and showing the 10 strongest of them, in the output grayscale image
%}
for labelVal = 1:N
	redIdx = idx{labelVal};
    greenIdx = idx{labelVal}+numRows*numCols;
    blueIdx = idx{labelVal}+2*numRows*numCols;
	outputImage(redIdx) = source(redIdx);
    outputImage(greenIdx) = source(greenIdx);
    outputImage(blueIdx) = source(blueIdx);
	points = detectSURFFeatures(rgb2gray(outputImage));
	%imshow(rgb2gray(outputImage)); hold on;
	imshow(outputImage); hold on;
	plot(points.selectStrongest(10));
end    

%-------------------------------------------------------------%
%Find Corresponding Points Between Two Images Using SURF Features
%Read images
img = imread('p1.jpg');
source1 = rgb2gray(img);
source2 = imread('pg.jpg');

%Detect SURF features
points1 = detectSURFFeatures(source1);
points2 = detectSURFFeatures(source2);

%Extract features
[f1, vpts1] = extractFeatures(source1, points1);
[f2, vpts2] = extractFeatures(source2, points2);

%Match features
indexPairs = matchFeatures(f1, f2) ;
matchedPoints1 = vpts1(indexPairs(:, 1));
matchedPoints2 = vpts2(indexPairs(:, 2));

%Visualize candidate matches
figure; ax = axes;
showMatchedFeatures(source1,source2,matchedPoints1,matchedPoints2,'Parent',ax);
title(ax, 'Putative point matches');
legend(ax,'Matched points 1','Matched points 2');

figure; ax = axes;
showMatchedFeatures(source1,source2,matchedPoints1,matchedPoints2,'montage','Parent',ax);
title(ax, 'Candidate point matches');
legend(ax, 'Matched points 1','Matched points 2');

%-------------------------------------------------------------%
%acquire image
source = imread("pg.jpg");

%Extract SURF features from an image
points = detectSURFFeatures(source);
[features, valid_points] = extractFeatures(source,points);

%Visualize 10 strongest SURF features, including their scales and orientation which were determined during the descriptor extraction process.
imshow(source); 
hold on;
strongestPoints = valid_points.selectStrongest(10);
strongestPoints.plot('showOrientation',true);