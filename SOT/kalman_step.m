function s = kalman_step(s, no_knowledge, global_params)
%KALMAN_STEP Summary of this function goes here
%   Detailed explanation goes here

R = global_params.R;
pattern = global_params.pattern;
H = global_params.H;

% If state hasn't been initialized
if isnan(s.x)
    
    % initialize state estimate from first observation
    %if diff(size(s.H))
    %error('Observation matrix must be square and invertible for state autointialization.');
    %end
    %s.x = inv(s.H)*s.z;
    %s.P = inv(s.H)*s.R*inv(s.H');
    
    
    %TODO better initial guess
    if isfield(s, 'z') && sum(isnan(s.z)) < 1
        n = size(s.z,1);
        s.x = [mean(s.z(1:n/3)); mean(s.z(n/3+1:2*n/3)); mean(s.z(2*n/3+1:n)); 0; 0; 0; 1];
        s.P = [ 30 0 0 0 0 0 0;
            0 30 0 0 0 0 0;
            0 0 30 0 0 0 0;
            0 0 0 1000 0 0 0;
            0 0 0 0 1000 0 0;
            0 0 0 0 0 1000 0;
            0 0 0 0 0 0 0];
    else
        % If we don't even have an observation, initialize at a random
        % position and zero velocity
        s.x = [rand; rand; rand; 0; 0; 0; 1];
        s.P = [ 100 0 0 0 0 0 0;
            0 100 0 0 0 0 0;
            0 0 100 0 0 0 0;
            0 0 0 1000 0 0 0;
            0 0 0 0 1000 0 0;
            0 0 0 0 0 1000 0;
            0 0 0 0 0 0 0];
    end
    
else
    % If we don't know which detections are missing, we need to come up
    % with a prediction for what detections are missing (the detections of which marker that is), i.e. we need to
    % find H and R which best explain the measurements.
    if no_knowledge &&  isfield(s,'z') && sum(isnan(s.z)) < 1
        detections = s.z;
        % Find an assignment between the individual markers and the detections
        assignment = match_patterns(pattern, reshape(detections, [],3));
        
        % Print in case an error was made in the assignment
        inversions = assignment(2:end) - assignment(1:end-1);
        if min(inversions) < 1
            assignment
        end
        % construct H and R from assignment vector, i.e. delete the
        % corresponding rows in H and R.
        s.H = H([assignment';assignment'+4; assignment'+8],:);
        s.R = R([assignment';assignment'+4; assignment'+8], [assignment';assignment'+4; assignment'+8]);
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

