function dist = distanceKalman(kalmanFilter, detections, motionType)
%DISTANCEKALMAN Summary of this function goes here
%   Detailed explanation goes here
[nDetections, dim] = size(detections);
if strcmp(kalmanFilter.type, 'LG-EKF')
    expectedMarkerLocations = measFuncNonHomogenous(kalmanFilter.mu, kalmanFilter.pattern);
else
    if strcmp(motionType, 'constAcc')
        expectedMarkerLocations = (Rot(kalmanFilter.x(3*3+1:3*3+4)) * kalmanFilter.pattern' + kalmanFilter.x(1:3))';
    else
        %fprintf('distanceKalman: no acc. used!')
        expectedMarkerLocations = (Rot(kalmanFilter.x(2*3+1:2*3+4)) * kalmanFilter.pattern' + kalmanFilter.x(1:3))';
    end
end
dist = pdist2(expectedMarkerLocations, detections);
%negLogL = zeros(4, nDetections);
%for i = 1:4
%    try
%        x = kalmanFilter.x;
%        f = kalmanFilter.J(x(7), x(8), x(9), x(10));
%        F = f(i:4:end,:);
%        covMat = F * kalmanFilter.P * F';
%        negLogL(i,:) = -reallog(mvnpdf(detections, expectedMarkerLocations(i,:), round(covMat,4)))';
%    catch
%        covMat
%    end
%end
end

