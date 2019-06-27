function dist = distanceKalman(kalmanFilter, detections)
%DISTANCEKALMAN Summary of this function goes here
%   Detailed explanation goes here
Rot = quatToMat();
expectedMarkerLocations = (Rot(kalmanFilter.x(2*3+1:2*3+4)) * kalmanFilter.pattern' + kalmanFilter.x(1:3))';
dist = pdist2(expectedMarkerLocations, detections);
end

