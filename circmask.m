function[m] = circmask(im,radius)
% Create a logical image of a circle with specified
% diameter, center, and image size.
% First create the image.
imageSizeX = size(im,1);
imageSizeY = size(im,2);
[columnsInImage rowsInImage] = meshgrid(1:imageSizeX, 1:imageSizeY);
% Next create the circle in the image.
centerX = imageSizeX/2;
centerY = imageSizeY/2;
m = (rowsInImage - centerY).^2 ...
    + (columnsInImage - centerX).^2 <= radius.^2;
% circlePixels is a 2D "logical" array.
end