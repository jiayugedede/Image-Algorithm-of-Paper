function output = multi(up, down, left, right, thresh)
%my version of the multi-flash edge detection
%pass in grayscale versions of the 4 images

t = 5;
if (nargin > 4)
    t = thresh;
end

height = size(up, 1);
width = size(up, 2);


%Capture ambient image I0
%Lets just use the average of the images as I0
ave = (double(up) + double(down) + double(left) + double(right)) / 4;

%find the Ik's
%can we get negative values here?

up_k = double(up) - ave;
down_k = double(down) - ave;
right_k = double(right) - ave;
left_k = double(left) - ave;

%of all of the Ik's, find the max at each pixel

Imax = max(max(up_k, down_k), max(right_k, left_k)); %this is like what they do...

%find the ratio images
%Rk = Ik/Imax

up_r = (up_k+5) ./ (Imax+5);
down_r = (down_k+5) ./ (Imax+5);
right_r = (right_k+5) ./ (Imax+5);
left_r = (left_k+5) ./ (Imax+5);

%Since we assume that the light sources lie directly on the XY axis through the
% center of the camera, the epipolar lines of each light source can be approximated by
% scanline or a column of the image.
%Iterate over all epipolar lines of the ratio images looking for large negative transitions

up_edge = zeros(height, width);
down_edge = zeros(height, width);
right_edge = zeros(height, width);
left_edge = zeros(height, width);

for i=2:(width-1),
    left_edge(:,i) = ((left_r(:,i) - left_r(:,i+1)) > t);
    right_edge(:,i) = ((right_r(:,i) - right_r(:,i-1)) > t);
end

for j=2:(height-1),
    up_edge(j,:) = ((up_r(j,:) - up_r(j-1,:)) > t);
    down_edge(j,:) = ((down_r(j,:) - down_r(j+1,:)) > t);
end

%all_edge = up_edge | down_edge | right_edge | left_edge;
output = up_edge | down_edge | right_edge | left_edge;