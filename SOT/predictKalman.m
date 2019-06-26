function [s, detectionsIdx] = predictKalman(s, noKnowledge, globalParams, model)
%PREDICTKALMAN Summary of this function goes here
%   Detailed explanation goes here

R = globalParams.R;
pattern = globalParams.pattern;
H = globalParams.H;

nMarkers = size(pattern,1);
dim = size(pattern,2);

switch model
    case 'linear'
        % If state hasn't been initialized
        if isnan(s.x)
            
            % initialize state estimate from first observation
            %if diff(size(s.H))
            %error('Observation matrix must be square and invertible for state autointialization.');
            %end
            %s.x = inv(s.H)*s.z;
            %s.P = inv(s.H)*s.R*inv(s.H');
            
            
            % TODO better initial guess
            % TODO how to initially guess the velocity?
            if isfield(s, 'z') && sum(isnan(s.z)) < 1
                detections = reshape(s.z, [], dim);
                assignment = match_patterns(pattern, detections, 'edges');
                positionEstimate = mean(detections - pattern(assignment,:),1);
                s.x = [positionEstimate'; zeros(dim,1); 1];
                s.P = eye(2*dim+1) .* repelem([globalParams.initPositionVar; globalParams.initMotionVar; 0], [dim, dim, 1]);
            else
                % If we don't even have an observation, initialize at a random
                % position and zero velocity
                s.x = [rand(dim,1); zeros(3,1); 1];
                s.P = eye(2*dim+1) .* repelem([globalParams.initMotionVar; globalParams.initMotionVar; 0], [dim, dim, 1]);
            end
            
        else
            % If we don't know which detections are missing, we need to come up
            % with a prediction for what detections are missing (the detections of which marker that is), i.e. we need to
            % find H and R which best explain the measurements.
            if noKnowledge &&  isfield(s,'z') && sum(isnan(s.z)) < 1
                detections = s.z;
                % Find an assignment between the individual markers and the detections
                assignment = match_patterns(pattern, reshape(detections, [],dim), 'ML');
                
                % Print in case an error was made in the assignment
                inversions = assignment(2:end) - assignment(1:end-1);
                if min(inversions) < 1
                    assignment
                end
                
                % construct H and R from assignment vector, i.e. delete the
                % corresponding rows in H and R.
                detectionsIdx = zeros(dim*length(assignment),1);
                for i = 1:dim
                    detectionsIdx( (i-1)*length(assignment) + 1: (i-1)*length(assignment) + length(assignment)) = assignment' + nMarkers*(i-1);
                end
                s.H = H(detectionsIdx,:);
                s.R = R(detectionsIdx, detectionsIdx);
            end
            
            % Here the actal kalman filter begins:
            
            % Prediction for state vector and covariance:
            s.x = s.A*s.x;
            s.P = s.A * s.P * s.A' + s.Q;
            % Compute Kalman gain factor:
            K = s.P*s.H'*inv(s.H*s.P*s.H'+s.R);
            
            % Correction based on observation (if observation is present):
            if isfield(s,'z') && sum(isnan(s.z)) < 1
                s.x = s.x + K*(s.z-s.H*s.x);
            end
            % TODO immer machen? oder nur wenn detection?
            % Correct covariance matrix estimate
            s.P = s.P - K*s.H*s.P;
        end
    case 'extended'
        % If state hasn't been initialized
        if isnan(s.x)
            
            % initialize state estimate from first observation
            %if diff(size(s.H))
            %error('Observation matrix must be square and invertible for state autointialization.');
            %end
            %s.x = inv(s.H)*s.z;
            %s.P = inv(s.H)*s.R*inv(s.H');
            
            
            % TODO better initial guess
            % TODO how to initially guess the velocity?
            if isfield(s, 'z') && sum(~isnan(s.z)) > 0
                detections = reshape(s.z, [], dim);
                Rot = quatToMat();
                assignment = match_patterns(pattern, detections, 'ML', Rot(0.25*ones(4,1)));
                positionEstimate = mean(detections - pattern(assignment,:),1);
                s.x = [positionEstimate'; zeros(dim,1); 0.25*ones(4,1); zeros(4,1)];
                s.P = eye(2*dim+8) .* repelem([globalParams.initPositionVar; globalParams.initMotionVar; globalParams.initQuatVar; globalParams.initQuatMotionVar], [dim, dim, 4, 4]);
                
                detectionsIdx = zeros(dim*length(assignment),1);
                for i = 1:dim
                    detectionsIdx( (i-1)*length(assignment) + 1: (i-1)*length(assignment) + length(assignment)) = assignment' + nMarkers*(i-1);
                end
                s.R = R(detectionsIdx, detectionsIdx);
            else
                % If we don't even have an observation, initialize at a random
                % position and zero velocity
                s.x = [rand(dim,1); zeros(3,1); 0.25*ones(4,1); zeros(4,1)];
                s.P = eye(2*dim+8) .* repelem([globalParams.initMotionVar],2*dim+8)';
            end
            
        else
            % If we don't know which detections are missing, we need to come up
            % with a prediction for what detections are missing (the detections of which marker that is), i.e. we need to
            % find H and R which best explain the measurements.
            if noKnowledge &&  isfield(s,'z') && sum(~isnan(s.z)) > 0
                detections = s.z;
                % Find an assignment between the individual markers and the detections
                Rot = quatToMat();
                assignment = match_patterns(pattern, reshape(detections, [],dim) - s.x(1:dim)', 'ML', Rot(s.x(2*dim+1:2*dim+4)));
                
                % Print in case an error was made in the assignment
                inversions = assignment(2:end) - assignment(1:end-1);
                if min(inversions) < 1
                    assignment
                end
                
                % construct H and R from assignment vector, i.e. delete the
                % corresponding rows in H and R.
                detectionsIdx = zeros(dim*length(assignment),1);
                for i = 1:dim
                    detectionsIdx( (i-1)*length(assignment) + 1: (i-1)*length(assignment) + length(assignment)) = assignment' + nMarkers*(i-1);
                end
                %s.H = @(x) s.H(x,assignment);
                s.R = R(detectionsIdx, detectionsIdx);
            end
             
            % Prediction for state vector and covariance:
            s.x = s.A*s.x;
            s.P = s.A * s.P * s.A' + s.Q;
        end
        
end


