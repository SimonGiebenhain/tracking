function s = kalman_step(s, no_knowledge, global_params)
%KALMAN_STEP Summary of this function goes here
%   Detailed explanation goes here

R = global_params.R;
pattern = global_params.pattern;
H = global_params.H;

num_markers = size(pattern,1);
dim = size(pattern,2);

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
        assignment = match_patterns(pattern, detections);
        position_estimate = mean(detections - pattern(assignment,:),1);
        s.x = [position_estimate'; zeros(dim,1); 1];   
        s.P = eye(2*dim+1) .* repelem([global_params.init_pos_var; global_params.init_motion_var; 0], [dim, dim, 1]);    
    else
        % If we don't even have an observation, initialize at a random
        % position and zero velocity
        s.x = [rand(dim,1); zeros(3,1); 1];
        s.P = eye(2*dim+1) .* repelem([global_params.init_motion_var; global_params.init_motion_var; 0], [dim, dim, 1]);  
    end
    
else
    % If we don't know which detections are missing, we need to come up
    % with a prediction for what detections are missing (the detections of which marker that is), i.e. we need to
    % find H and R which best explain the measurements.
    if no_knowledge &&  isfield(s,'z') && sum(isnan(s.z)) < 1
        detections = s.z;
        % Find an assignment between the individual markers and the detections
        assignment = match_patterns(pattern, reshape(detections, [],dim));
        
        % Print in case an error was made in the assignment
        inversions = assignment(2:end) - assignment(1:end-1);
        if min(inversions) < 1
            assignment
        end
        
        % construct H and R from assignment vector, i.e. delete the
        % corresponding rows in H and R.
        detections_idx = zeros(dim*length(assignment),1);
        for i = 1:dim
            detections_idx( (i-1)*length(assignment) + 1: (i-1)*length(assignment) + length(assignment)) = assignment' + num_markers*(i-1);
        end
        s.H = H(detections_idx,:);
        s.R = R(detections_idx, detections_idx);
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

end

