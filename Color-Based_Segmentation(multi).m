%acquire images
source1 = imread("p1.jpg");
source2 = imread("p2.jpg");
source3 = imread("p3.jpg");
subplot(1,3,1), imshow(source1);
title("Parthenon 1");
subplot(1,3,2), imshow(source2);
title("Parthenon 2");
subplot(1,3,3), imshow(source3);
title("Parthenon 3");

%Calculate Sample Colors in L*a*b* Color Space for Each Region
%{
You can see six major colors in the image: the background color, red, green, purple, yellow, and magenta. 
The L*a*b* colorspace (also known as CIELAB or CIE L*a*b*) enables you to quantify these visual differences.

The L*a*b* color space is derived from the CIE XYZ tristimulus values. The L*a*b* space consists of a luminosity 'L*' 
or brightness layer, chromaticity layer 'a*' indicating where color falls along the red-green axis, and chromaticity layer 'b*' 
indicating where the color falls along the blue-yellow axis.

Your approach is to choose a small sample region for each color and to calculate each sample region's average color in 'a*b*' space. 
You will use these color markers to classify each pixel.
%}

load regioncoordinates;

nColors = 6;
sample_regions1 = false([size(source1,1) size(source1,2) nColors]);
sample_regions2 = false([size(source2,1) size(source2,2) nColors]);
sample_regions3 = false([size(source3,1) size(source3,2) nColors]);

for count = 1:nColors
  sample_regions(:,:,count) = roipoly(source,region_coordinates(:,1,count), ...
                                      region_coordinates(:,2,count));
end

imshow(sample_regions(:,:,2))
title('Sample Region for Red')

%Convert your source RGB image into an L*a*b* image using rgb2lab
labImage = rgb2lab(source);

%Calculate the mean 'a*' and 'b*' value for each area that you extracted with roipoly. 
%These values serve as your color markers in 'a*b*' space
AImage = labImage(:, :, 2);
BImage = labImage(:, :, 3);

color_markers = zeros([nColors, 2]);

for count = 1:nColors
  color_markers(count,1) = mean2(AImage(sample_regions(:,:,count)));
  color_markers(count,2) = mean2(BImage(sample_regions(:,:,count)));
end

%Example the average color of the red sample region in 'a*b*' space is:
fprintf('[%0.3f,%0.3f] \n',color_markers(2,1),color_markers(2,2));

%Classify Each Pixel Using the Nearest Neighbor Rule
%{
Each color marker now has an 'a*' and a 'b*' value. You can classify each pixel in the lab_fabric image by calculating the Euclidean distance 
between that pixel and each color marker. The smallest distance will tell you that the pixel most closely matches that color marker. 
For example, if the distance between a pixel and the red color marker is the smallest, then the pixel would be labeled as a red pixel.

Create an array that contains your color labels, i.e., 0 = background, 1 = red, 2 = green, 3 = purple, 4 = magenta, and 5 = yellow.
%}
color_labels = 0:nColors-1;

%Initialize matrices to be used in the nearest neighbor classification
AImage = double(AImage);
BImage = double(BImage);
distance = zeros([size(AImage), nColors]);

%Perform classification
for count = 1:nColors
  distance(:,:,count) = ( (AImage - color_markers(count,1)).^2 + ...
                      (BImage - color_markers(count,2)).^2 ).^0.5;
end

[~,label] = min(distance,[],3);
label = color_labels(label);
clear distance;

%Display Results of Nearest Neighbor Classification
%{
The label matrix contains a color label for each pixel in the source image. Use the label matrix to separate objects in the original 
source image by color. Display the five segmented colors as a montage. Also display the background pixels in the image that are not 
classified as a color.
%}
rgb_label = repmat(label,[1 1 3]);
segmented_images = zeros([size(source), nColors],'uint8');

for count = 1:nColors
  color = source;
  color(rgb_label ~= color_labels(count)) = 0;
  segmented_images(:,:,:,count) = color;
end 

montage({segmented_images(:,:,:,2),segmented_images(:,:,:,3) ...
    segmented_images(:,:,:,4),segmented_images(:,:,:,5) ...
    segmented_images(:,:,:,6),segmented_images(:,:,:,1)});
title("Montage of Red, Green, Purple, Magenta, and Yellow Objects, and Background")

%Display 'a*' and 'b*' Values of the Labeled Colors
%{
You can see how well the nearest neighbor classification separated the different color populations by plotting the 'a*' and 'b*' values 
of pixels that were classified into separate colors. For display purposes, label each point with its color label.
%}
purple = [119/255 73/255 152/255];
plot_labels = {'k', 'r', 'g', purple, 'm', 'y'};

figure
for count = 1:nColors
  plot(AImage(label==count-1),BImage(label==count-1),'.','MarkerEdgeColor', ...
       plot_labels{count}, 'MarkerFaceColor', plot_labels{count});
  hold on;
end
  
title('Scatterplot of the segmented pixels in ''a*b*'' space');
xlabel('''a*'' values');
ylabel('''b*'' values');

%--------------------------------------------------------------------------%
% Read the image and convert to L*a*b* color space
I = imread('vegetables.jpg');
Ilab = rgb2lab(I);
% Extract a* and b* channels and reshape
ab = double(Ilab(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);
% Segmentation usign k-means
nColors = 4;
[cluster_idx, cluster_center] = kmeans(ab,nColors,...
  'distance',     'sqEuclidean', ...
  'Replicates',   3);
% Show the result
pixel_labels = reshape(cluster_idx,nrows,ncols);
imshow(pixel_labels,[]), title('image labeled by cluster index')
