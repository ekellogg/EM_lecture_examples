function[normim] = normalize_image(im)
    normim = im - min(im(:)) ./ max(im(:));
end