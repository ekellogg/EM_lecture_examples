function[im] = showImage(im)
    minVal = min(min(im));
    im = im - minVal;
    maxVal = max(max(im));
    im = im / maxVal * 100;
    image( im );
    
end