%acquire image
source = imread("vegetables.jpg");

%Convert your source RGB image into an L*a*b* image using rgb2lab
labImage = rgb2lab(source);

%Get each channel 'L*', 'a*' and 'b*' for the L*a*b* image
LImage = labImage(:, :, 1);
AImage = labImage(:, :, 2);
BImage = labImage(:, :, 3);

%Show the L*a*b* image
subplot(4, 2, 1.5);
imshow(labImage);
title('L*a*b* Image', 'FontSize', 15);

%Show each of the channels individually 
%and by scaling the display based on the range of pixel values
subplot(4, 2, 3);
imshow(LImage);
title('L channel Image', 'FontSize', 15);
subplot(4, 2, 4);
imshow(LImage, []);
title('L channel scaled Image', 'FontSize', 15);
subplot(4, 2, 5);
imshow(AImage);
title('A channel Image', 'FontSize', 15);
subplot(4, 2, 6);
imshow(AImage, []);
title('A channel scaled Image', 'FontSize', 15);
subplot(4, 2, 7);
imshow(BImage);
title('B channel Image', 'FontSize', 15);
subplot(4, 2, 8);
imshow(BImage, []);
title('B channel scaled Image', 'FontSize', 15);