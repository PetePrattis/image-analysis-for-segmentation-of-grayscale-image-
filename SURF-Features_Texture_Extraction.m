%acquire images
source1 = imread("vg.jpg");
img = imread("v1.jpg");
source2 = rgb2gray(img);

[L1,N1] = superpixels(source1,100);
[L2,N2] = superpixels(source2,100);

BW1 = boundarymask(L1);
BW2 = boundarymask(L2);

outputImage1 = zeros(size(source1), 'like', source1);
outputImage2 = zeros(size(img), 'like', img);

%{
I'll use the function label2idx to compute the indices of the pixels in each superpixel cluster. 
That will let me access the red, green, and blue component values using linear indexing
%}
idx1 = label2idx(L1);

idx2 = label2idx(L2);
numRows = size(img,1);
numCols = size(img,2);

%{
For each of the N superpixel clusters, use linear indexing reconstruct the corresponding pixels, 
while detecting/extracting SURF Features and showing the 10 strongest of them, in the grayscale image
%}
for labelVal1 = 1:N1
	Idx = idx1{labelVal1};
	outputImage1(Idx) = source1(Idx);
	points1 = detectSURFFeatures(outputImage1);
	[f1, vpts1] = extractFeatures(outputImage1, points1);
	figure(1);
	imshow(outputImage1); hold on;
	strongestPoints1 = points1.selectStrongest(10);
	strongestPoints1.plot('showOrientation',true);
	grid;
end

%{
For each of the N superpixel clusters, use linear indexing to access the red, green, and blue components, 
reconstruct the corresponding pixels while detecting/extracting SURF Features for the grayscale version of the image
and showing the 10 strongest of them, in the output grayscale image
%}
for labelVal = 1:N2
	redIdx = idx2{labelVal};
    greenIdx = idx2{labelVal}+numRows*numCols;
    blueIdx = idx2{labelVal}+2*numRows*numCols;
	outputImage2(redIdx) = img(redIdx);
    outputImage2(greenIdx) = img(greenIdx);
    outputImage2(blueIdx) = img(blueIdx);
	points2 = detectSURFFeatures(rgb2gray(outputImage2));
	[f2, vpts2] = extractFeatures(rgb2gray(outputImage2), points2);
	figure(2);
	%imshow(img); hold on;
	imshow(outputImage2); hold on;
	strongestPoints2 = points2.selectStrongest(10);
	strongestPoints2.plot('showOrientation',true);
	grid;
end 

%Match features both ways
indexPairs1 = matchFeatures(f1, f2);
indexPairs2 = matchFeatures(f2, f1);
matchedPoints1 = vpts1(indexPairs1(:, 1));
matchedPoints2 = vpts2(indexPairs1(:, 2));
matchedPoints3 = vpts1(indexPairs2(:, 2));
matchedPoints4 = vpts2(indexPairs2(:, 1));

%Visualize candidate matches
figure; ax = axes;
showMatchedFeatures(source1,source2,matchedPoints1,matchedPoints2,'Parent',ax);
showMatchedFeatures(source1,source2,matchedPoints3,matchedPoints4,'Parent',ax);
title(ax, 'Putative point matches');
legend(ax,'Matched points 1','Matched points 2');

figure; ax = axes;
showMatchedFeatures(source1,source2,matchedPoints1,matchedPoints2,'montage','Parent',ax);
showMatchedFeatures(source1,source2,matchedPoints3,matchedPoints4,'montage','Parent',ax);
title(ax, 'Candidate point matches');
legend(ax, 'Matched points 1','Matched points 2');