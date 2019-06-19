function [s, global_params] = setup_kalman(pattern, T, model, obs_noise, process_noise)
%SETUP_KALMAN Summary of this function goes here
%   Detailed explanation goes here

dim = size(pattern,2);
num_markers = size(pattern,1);

switch model
    % linear klaman filter with constant velocity
    case 'linear'
        % pre-allocate struct array (fields are not pre-allocated)
        % Altough the space for the single fileds is not pre-allocated, empirically it showed that this has
        % benefits for large T
        
        state_dim = 2*dim+1;
        observation_dim = num_markers*dim;
        
        s(T+2).A = zeros(state_dim);
        s(T+2).Q = zeros(state_dim);
        s(T+2).H = zeros(observation_dim,state_dim);
        s(T+2).R = zeros(observation_dim);
        s(T+2).x = zeros(state_dim,1);
        s(T+2).P = zeros(state_dim);
        
        % Set up dynamics of the system and model assumptions
        
        % Transition matrix.
        % Implements this simple relationship: position_{t+1} = position_t + velocity_t
        s(1).A = eye(state_dim);
        for i = 1:dim
            s(1).A(i,dim+i) = 1;
        end
        
        % The process covariance matrix
        s(1).Q = process_noise*eye(state_dim);
        s(1).Q(state_dim,state_dim) = 0; % The last component of the state has to stay at 1
        
        % The observation matrix
        H = [ repelem(eye(dim),num_markers,1) zeros(observation_dim, dim) pattern(:)];
        s(1).H = H;
        
        % The measurment covariance matrix
        R = obs_noise*eye(observation_dim);
        s(1).R = R;
        
        global_params.R = R;
        global_params.H = H;
        global_params.pattern = pattern;
        
        % Do not specify an initial state
        s(1).x = nan;
        s(1).P = nan;
        
    case 'extended'
        error('TODO')
        
        
end

