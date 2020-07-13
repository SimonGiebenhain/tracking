function [newTrack] = createLGEKFtrack(rotm, pos, l2Error, patternIdx, pattern, patternName, params, motionModel, ghostKF)
%traLGEKFTRACK This function constructs a struct holding all relevant
%information of a LG-EKF.
%   Arguments:
%   @rotm [3x3] rotation matrix, describing the estimated current
%   orientation of the bird
%   @pos 3-d vector descriping the estimated current position of the bird
%   @l2Error scalar indicating how well rotm and pos explain the current
%   measurment. The higher l2Error is, the higher will be the initial
%   uncertainty of the state in the LG-EKF.
%   @patternIdx
%   @pattern
%   @patternName
%   @params
%   @motionModel
%   @ghostKf

if ~exist('motionModel', 'var') || motionModel == -1
    mM = params.initMotionModel;
else
    if exist('ghostKF', 'var')
        if length(ghostKF.x) > 3 && norm(ghostKF.x(4:6)) > 5
            mM = 2;
        else
            mM = 0; 
        end      
    else
        mM = motionModel;
    end
end
%if exist('ghostKF', 'var')
%    mM = 2;
%end

[s, kalmanParams] = setupKalman(pattern, -1, params, mM);
mu.X = [rotm       pos; 
        zeros(1,3) 1   ];
%if exist('ghostKF', 'var')
%    mu.v = ghostKF.x(4:6);
%    mM = ghostKF.x(7:9);
%else
%    mu.v = zeros(3,1);
%    mu.a = zeros(3,1);
%end
mu.motionModel = mM;
certaintyFactor = min(l2Error^2, 1);
if mM == 0
    s.P = diag(repelem([certaintyFactor*params.initialNoise.initQuatVar;
    certaintyFactor*params.initialNoise.initPositionVar;
    ],[3, 3]));
    s.mu = mu;
elseif mM == 1
    s.P = diag(repelem([certaintyFactor*params.initialNoise.initQuatVar;
    certaintyFactor*params.initialNoise.initPositionVar;
    params.initialNoise.initMotionVar],[3, 3, 3]));
    s.mu = mu;
elseif mM == 2
    if exist('ghostKF', 'var')
        if isfield(ghostKF, 'isFake')
            mu.v = ghostKF.v;
            mu.a = zeros(3,1);
            s.P = diag(repelem([certaintyFactor*params.initialNoise.initQuatVar;
                certaintyFactor*params.initialNoise.initPositionVar;
                params.initialNoise.initMotionVar;
                params.initialNoise.initAccVar
                ],[3, 3, 3, 3]));
        else
            mu.v = ghostKF.x(4:6)/2;
            mu.a = zeros(3,1);%ghostKF.x(7:9)/2;
            s.P = diag(repelem([certaintyFactor*params.initialNoise.initQuatVar;
                    certaintyFactor*params.initialNoise.initPositionVar;
                    params.initialNoise.initMotionVar;
                    params.initialNoise.initAccVar
                    ],[3, 3, 3, 3]));
            s.P(4:end, 4:end) = ghostKF.P;
        end
    else
        mu.v = zeros(3,1);
        mu.a = zeros(3,1);
        s.P = diag(repelem([certaintyFactor*params.initialNoise.initQuatVar;
                certaintyFactor*params.initialNoise.initPositionVar;
                params.initialNoise.initMotionVar;
                params.initialNoise.initAccVar
                ],[3, 3, 3, 3]));
    end
    s.mu = mu;
end

s.pattern = pattern;
s.flying = -1;
s.consecutiveInvisibleCount = 0;
s.framesInNewMotionModel = 11;
s.latest5pos = zeros(5, 3);
s.latest5pos(1, :) = pos;
s.latestPosIdx = 1;
newTrack = struct(...
    'id', patternIdx, ...
    'name', patternName, ...
    'kalmanFilter', s, ...
    'kalmanParams', kalmanParams, ...
    'age', 1, ...
    'totalVisibleCount', 1, ...
    'consecutiveInvisibleCount', 0);
end

