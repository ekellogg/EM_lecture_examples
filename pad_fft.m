function[newim] = pad_fft(img,scale)
    fftnewim = zeros(size(img)*scale);
    c = size(fftnewim)./2;
    o = size(img);
    %the indices really seem to matter wrt the fft padding!! be careful..
    s = c - o/2 + 1;
    e = c + o/2;
    fftnewim( (s(1)):(e(1)), (s(2)):(e(2)) ) = fftshift(fft2(img));
    newim = real(ifft2(ifftshift( fftnewim )));
  
end