function [s, globalParams] = setupKalman(pattern, T, parameters, patternName)
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
model = parameters.model;
quatMotionType = parameters.quatMotionType;
motionType = parameters.motionType;
obsNoise = parameters.measurementNoise;
processNoise = parameters.processNoise;
initialNoise = parameters.initialNoise;
%patternName = parameters.patternName;
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
            if strcmp(motionType, 'constAcc')
                stateDim = 3*dim + 4;
            else
                stateDim = 2*dim + 4;
            end
        else
            if strcmp(motionType, 'constAcc')
                %TODO
                fprintf('notimplementedyet')
            else
                stateDim = 2*dim+8;
            end
        end
        obsDim = nMarkers*dim;
        
        s(T+2).A = zeros(stateDim);
        s(T+2).Q = zeros(stateDim);
        s(T+2).H = zeros(obsDim, stateDim);
        s(T+2).R = zeros(obsDim);
        s(T+2).x = zeros(stateDim, 1);
        s(T+2).P = zeros(stateDim);
        %s(T+2).assignemnt = 
        
        % Set up dynamics of the system and model assumptions
        
        % State Transition matrix A:
        % Implements this simple relationship: position_{t+1} = position_t + velocity_t
        % or if constAcc model is used this:   position_{t+1} = position_t
        % + velocity_t + 1/2*acceleration_t
        s(1).A = eye(stateDim);
        for i = 1:1*dim
            s(1).A(i,dim+i) = 1;
        end
        if strcmp(motionType, 'constAcc')
            for i = dim+1:2*dim
                s(1).A(i,dim+i) = 1;
            end
            for i=1:dim
                s(1).A(i, 2*dim+i) = 1/2;
            end
        end
        
        if ~strcmp(quatMotionType, 'brownian')
            for i =1:4
                s(1).A(2*dim+i,2*dim+4+i) = 1;
            end
        end
        
        % The process covariance matrix
        if strcmp(quatMotionType, 'brownian')
            if strcmp(motionType, 'constAcc')
                processNoiseScale = [ repmat(processNoise.position,dim,1); repmat(processNoise.motion,dim,1); ...
                                      repmat(processNoise.acceleration,dim,1); ...
                                      repmat(processNoise.quat,4,1)];
            else
                processNoiseScale = [ repmat(processNoise.position,dim,1); repmat(processNoise.motion,dim,1); ...
                                      repmat(processNoise.quat,4,1)];
            end
        else
            if strcmp(motionType, 'constAcc')
                fprintf('not implemented yet')
            else
                processNoiseScale = [ repmat(processNoise.position,dim,1); repmat(processNoise.motion,dim,1);...
                                      repmat(processNoise.quat,4,1); repmat(processNoise.quatMotion,4,1) ];
            end
        end
        s(1).Q = processNoiseScale .* eye(stateDim);
        
        % get the measurement function and its jacobian, both are function
        % handles and need to be supplied with a quaternion 
        if strcmp(motionType, 'constAcc')
            J = @(q1, q2, q3, q4) EKFJacConstAcc(pattern(1,1), pattern(1,2), pattern(1,3),pattern(2,1),pattern(2,2),pattern(2,3),pattern(3,1),pattern(3,2),pattern(3,3),pattern(4,1),pattern(4,2),pattern(4,3),q1, q2, q3, q4);
            H = @(xvec) reshape( (Rot(xvec(3*3+1:3*3+4)) * pattern')' + xvec(1:3)', [], 1 );

        else
            J = @(q1, q2, q3, q4) EKFJac(pattern(1,1), pattern(1,2), pattern(1,3),pattern(2,1),pattern(2,2),pattern(2,3),pattern(3,1),pattern(3,2),pattern(3,3),pattern(4,1),pattern(4,2),pattern(4,3),q1, q2, q3, q4);
            H = @(xvec) reshape( (Rot(xvec(2*3+1:2*3+4)) * pattern')' + xvec(1:3)', [], 1 );

        end

        %[H, J] = getMeasurementFunction(pattern, quatMotionType, motionType);
        
        
        s(1).H = H;
        s(1).J = J;
        %s(1).J = str2func(prefix);
        % The measurment covariance matrix
        R = obsNoise*eye(obsDim);
        s(1).R = R;
        s(1).type = 'EKF';
        %s(1).patternName = patternName;
        
        
        globalParams.R = R;
        globalParams.H = H;
        globalParams.pattern = pattern;
        globalParams.obsNoise = obsNoise;
        globalParams.processNoise = processNoise;
        globalParams.initPositionVar   = initialNoise.initPositionVar;
        globalParams.initMotionVar     = initialNoise.initMotionVar;
        globalParams.initAccVar        = initialNoise.initAccVar;
        globalParams.initQuatVar       = initialNoise.initQuatVar;
        globalParams.initQuatMotionVar = initialNoise.initQuatMotionVar;
        
        
        
        % Do not specify an initial state
        s(1).x = nan;
        s(1).P = nan;
        
    case 'LieGroup'      
        
        mM = 0;
        % working in homogenous coordinates, hence dim + 1 is used
        obsDim = nMarkers*(dim+1);
                   
        % The process covariance matrix
        if mM == 0
            s.Q = diag([repmat(processNoise.quat, dim, 1);
                    repmat(processNoise.position, dim, 1)]);
        elseif mM == 1
            s.Q = diag([repmat(processNoise.quat, dim, 1);
                    repmat(processNoise.position, dim, 1);
                    repmat(processNoise.motion, dim, 1)]);
        elseif mM == 2
            s.Q = diag([repmat(processNoise.quat, dim, 1);
                    repmat(processNoise.position, dim, 1);
                    repmat(processNoise.motion, dim, 1);
                    repmat(processNoise.acceleration, dim, 1)]);
        end
        
        

        % The measurment covariance matrix
        R = obsNoise*eye(obsDim);
        globalParams.R = R;
        globalParams.pattern = pattern;

        % Do not specify an initial state
        mu.X = nan;
        mu.v = nan;
        mu.a = nan;
        s.mu = mu;
        s.P = nan;
        s.type = 'LG-EKF';
        %s.patternName = patternName;
end
end


