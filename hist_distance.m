clear;
clc;

addrr= "directory path of generated images";
gts = "directory path of ground-truth images";

%number of test images, 90 for UIEB and 515 for EUVP
num_test = 0;
distances = zeros(num_test,2);

filenames = dir(gts);
filenames = filenames(3:end,:);


filenames2 = dir(addrr);
filenames2 = filenames2(3:end,:);
for i=1:num_test
    im1 = rgb2hsv(double(imread(strcat(addrr,filenames2(i).name))));
    im2 = rgb2hsv(double(imread(strcat(gts,filenames(i).name))));

    dist = 0;
    for j=1:3
        hist_pred = (imhist(im1(:,:,j))+eps);
        hist_pred=hist_pred./sum(hist_pred);
        hist_gt = (imhist(im2(:,:,j))+eps);
        hist_gt = hist_gt./sum(hist_gt);
        dist = sum((hist_pred-hist_gt).^2);
        distances(i,j) = chi_square_statistics(hist_pred',hist_gt');
    end
end
mean(distances)
return