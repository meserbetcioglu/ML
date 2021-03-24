clear, clc, close all;

nasacolor = imread('TarantulaNebula.jpg');

figure
image(nasacolor)
title('Original Image')

nasa = sum(nasacolor, 3, 'double');
m = max(max(nasa));
nasa = nasa*255/m;

figure
image(nasa)
title('Grayscale NASA photo (without colormap)')

figure
colormap(gray(256));
image(nasa)
title('Grayscale NASA photo');

[U S V] = svd(nasa);

%Max value is 61930, values fall to less than %2 in after 38 points. This
%shows that among 567 basis vectors, only 38 of them contribute more than
%two percent of what biggest vector contributes to the picture.
figure
semilogy(diag(S)) 

nasa100=U(:,1:100)*S(1:100,1:100)*V(:,1:100)';
nasa50=U(:,1:50)*S(1:50,1:50)*V(:,1:50)';
nasa25=U(:,1:25)*S(1:25,1:25)*V(:,1:25)';

%Data is much smaller than before with 25 basis vectors. 
%The details are missing, image is blurry and grainy. Although the image
%isn't very clear, it is still recognizable.
figure
colormap(gray(256));
image(nasa25)
title('Grayscale Reconstructed NASA Photo (25 Basis Vectors)')

%With 25 more basis vectors, image becomes much clearer. Most of the
%details are in and image is very recognizable. Data site doubled compared
%to the one before but it is still very small compared to the original.
figure
colormap(gray(256));
image(nasa50)
title('Grayscale Reconstructed NASA Photo (50 Basis Vectors)')

%With 100 basis vectors, the image is almost the same as the original.
%There are still tiny errors but those are unrecognizable without through
%examination. Almost all details are in, even very small stars. 
%This reconstructed image is still more than 5 times smaller 
%than the original.
figure
colormap(gray(256));
image(nasa100)
title('Grayscale Reconstructed NASA Photo (100 Basis Vectors)')
