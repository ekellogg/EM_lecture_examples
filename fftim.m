function[F] = fftim(im)
    F = fftshift( abs( fft2( im )));
    %F = fftshift(fft2(im)); %??
end