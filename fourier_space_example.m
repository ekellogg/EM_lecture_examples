%read in image and display
img = ReadMRC('pusheen.mrc');
colormap(bone(100)); %make grayscale
showImage(img);

%show fourier transform of cat
fft_cat = fftshift(fft2(img));
showImage(log(abs(fft_cat)));

%mask high-resolution fourier space (i.e. lowpass filter)
%create a gaussian mask that keeps low-resolution information%
m=gauss2d(img,100,size(img)./2);
showImage(m);
masked_fft_cat = m.*(fft_cat);
lowpass_cat = abs(ifft2(ifftshift( masked_fft_cat )));
showImage(lowpass_cat);

%high pass cat
m=gauss2d(img,10000,size(img)./2);
m=abs(1-m);
showImage(m);
masked_fft_cat = m.*(fft_cat);
lowpass_cat = abs(ifft2(ifftshift( masked_fft_cat )));
showImage(lowpass_cat);
