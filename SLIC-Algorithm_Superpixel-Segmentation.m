%Convert your source RGB image into an L*a*b* image using rgb2lab
labImage = rgb2lab(source);

B = superpixels(labImage,100,'IsInputLab',true, 'Method','slic')
bw = boundarymask(B);
imshow(imoverlay(source,bw,'cyan'),'InitialMagnification',67);