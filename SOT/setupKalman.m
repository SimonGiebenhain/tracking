function [s, globalParams] = setupKalman(pattern, T, model, quatMotionType, obsNoise, processNoise, initialNoise)
%SETUPKALMAN This function initializes all necessary parameters and
%functions for performing tracking with the kalman filter.
%
%   @pattern is the pattern of markers for this object
%
%   @T is the number of frames
%
%   @model 'linear' means the regulat kalman filter is prepared. 'extended'
%       means that the extended klaman filter (EKF) is prepared.
%
%   @obsNoise The variance of the noise contained in the measurements
%
%   @processNoise A struct containing the variances of the different
%       components of the state, i.e. position, velocity_position,
%       quaternion, velocity_quaternion
%
% The noise parameters are important hyperparameters and need to roughly
% agree with the actual noise in the data for good performance.

dim = size(pattern,2);
nMarkers = size(pattern,1);

switch model
    % linear klaman filter with constant velocity
    case 'linear'
        % pre-allocate struct array (fields are not pre-allocated)
        % Altough the space for the single fileds is not pre-allocated, empirically it showed that this has
        % benefits for large T
        
        stateDim = 2*dim+1;
        obsDim = nMarkers*dim;
        
        s(T+2).A = zeros(stateDim);
        s(T+2).Q = zeros(stateDim);
        s(T+2).H = zeros(obsDim,stateDim);
        s(T+2).R = zeros(obsDim);
        s(T+2).x = zeros(stateDim,1);
        s(T+2).P = zeros(stateDim);
        
        % Set up dynamics of the system and model assumptions
        
        % Transition matrix.
        % Implements this simple relationship: position_{t+1} = position_t + velocity_t
        s(1).A = eye(stateDim);
        for i = 1:dim
            s(1).A(i,dim+i) = 1;
        end
        
        % The process covariance matrix
        processNoiseScale = [repmat(processNoise.position,dim,1); repmat(processNoise.motion,dim,1); 0];
        s(1).Q = processNoiseScale .* eye(stateDim);
        
        % The observation matrix
        H = [ repelem(eye(dim),nMarkers,1) zeros(obsDim, dim) pattern(:)];
        s(1).H = H;
        
        % The measurment covariance matrix
        R = obsNoise*eye(obsDim);
        s(1).R = R;
        
        globalParams.R = R;
        globalParams.H = H;
        globalParams.pattern = pattern;
        
        % Do not specify an initial state
        s(1).x = nan;
        s(1).P = nan;
        
    case 'extended'
        % pre-allocate struct array (fields are not pre-allocated)
        % Altough the space for the single fileds is not pre-allocated, empirically it showed that this has
        % benefits for large T
        
        if strcmp(quatMotionType, 'brownian')
            stateDim = 2*dim + 4;
        else
            stateDim = 2*dim+8;
        end
        obsDim = nMarkers*dim;
        
        s(T+2).A = zeros(stateDim);
        s(T+2).Q = zeros(stateDim);
        s(T+2).H = zeros(obsDim, stateDim);
        s(T+2).R = zeros(obsDim);
        s(T+2).x = zeros(stateDim, 1);
        s(T+2).P = zeros(stateDim);
        
        % Set up dynamics of the system and model assumptions
        
        % Transition matrix.
        % Implements this simple relationship: position_{t+1} = position_t + velocity_t
        s(1).A = eye(stateDim);
        for i = 1:dim
            s(1).A(i,dim+i) = 1;
        end
        
        if ~strcmp(quatMotionType, 'brownian')
            for i =1:4
                s(1).A(2*dim+i,2*dim+4+i) = 1;
            end
        end
        
        % The process covariance matrix
        if strcmp(quatMotionType, 'brownian')
            processNoiseScale = [ repmat(processNoise.position,dim,1); repmat(processNoise.motion,dim,1);...
                                    repmat(processNoise.quat,4,1)];
        else
            processNoiseScale = [ repmat(processNoise.position,dim,1); repmat(processNoise.motion,dim,1);...
                                    repmat(processNoise.quat,4,1); repmat(processNoise.quatMotion,4,1) ];
        end
        s(1).Q = processNoiseScale .* eye(stateDim);
        
        % get the measurement function and its jacobian, both are function
        % handles and need to be supplied with a quaternion
        [H, J] = getMeasurementFunction(pattern, quatMotionType);
        
        s(1).H = H;
        s(1).J = J;
        % The measurment covariance matrix
        R = obsNoise*eye(obsDim);
        s(1).R = R;
        
        globalParams.R = R;
        globalParams.H = H;
        globalParams.pattern = pattern;
        globalParams.obsNoise = obsNoise;
        globalParams.processNoise = processNoise;
        globalParams.initPositionVar   = initialNoise.initPositionVar;
        globalParams.initMotionVar     = initialNoise.initMotionVar;
        globalParams.initQuatVar       = initialNoise.initQuatVar;
        globalParams.initQuatMotionVar = initialNoise.initQuatMotionVar;
        
        
        
        % Do not specify an initial state
        s(1).x = nan;
        s(1).P = nan;
end
end


