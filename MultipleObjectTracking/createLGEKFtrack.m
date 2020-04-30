function [newTrack] = createLGEKFtrack(rotm, pos, l2Error, patternIdx, pattern, patternName, params)
%CREATELGEKFTRACK Summary of this function goes here
%   Detailed explanation goes here

[s, kalmanParams] = setupKalman(pattern, -1, params);
mu.X = [rotm pos; zeros(1,3) 1];
mu.v = zeros(3,1);
mu.a = zeros(3,1);
s.mu = mu;
certaintyFactor = min(l2Error^2, 1);
s.P = diag(repelem([certaintyFactor*params.initialNoise.initQuatVar;
    certaintyFactor*params.initialNoise.initPositionVar;
    params.initialNoise.initMotionVar;
    params.initialNoise.initAccVar
    ],[3, 3, 3, 3]));
s.pattern = pattern;
s.flying = -1;
s.consecutiveInvisibleCount = 0;
newTrack = struct(...
    'id', patternIdx, ...
    'name', patternName, ...
    'kalmanFilter', s, ...
    'kalmanParams', kalmanParams, ...
    'age', 1, ...
    'totalVisibleCount', 1, ...
    'consecutiveInvisibleCount', 0);
end

