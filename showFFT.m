function[sum_pw] = showFFT(im,bxsz)
    % subdivide image into boxes
    % compute power spectrum of each box and sum individual spectra to
    % obtain final working power spectrum
    box_size = 0;
    if(nargin == 1)
        box_size=512;
    else
        box_size=bxsz;
    end
    im_size = size(im,1) %assume square
    sum_pw = zeros(box_size,box_size);
    num_box = 0;
    for(i = 1:box_size:(im_size-box_size))
        for(j = 1:box_size:(im_size-box_size))
            boxed_im = im(i:(i+box_size-1),j:(j+box_size-1));
            F = fftim(boxed_im);
            sum_pw = sum_pw + F;
        end
    end
    showImage(log(sum_pw));
end