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
        process_noise_scale = [repmat(process_noise.pos,dim,1); repmat(process_noise.motion,dim,1); 0];
        s(1).Q = process_noise_scale .* eye(state_dim);
        
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
        % pre-allocate struct array (fields are not pre-allocated)
        % Altough the space for the single fileds is not pre-allocated, empirically it showed that this has
        % benefits for large T
        
        state_dim = 2*dim+8;
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
        for i =1:4
            s(1).A(2*dim+i,2*dim+4+i) = 1;
        end
        
        % The process covariance matrix
        process_noise_scale = [ repmat(process_noise.pos,dim,1); repmat(process_noise.motion,dim,1);...
            repmat(process_noise.quat_mot,4,1); repmat(process_noise.quat_mot,4,1) ];
        s(1).Q = process_noise_scale .* eye(state_dim);
        
        % The observation function
        [H, J] = getMeasurementFunction(pattern);
        % Rot = subs(Rot_sym, [q1 q2 q3 q4], x(2*dim+1:2*dim+4));
        %H = @(x, observation_idx) reshape( (subs(quat_to_mat(x(2*dim+1:2*dim+4)),[q1 q2 q3 q4], x(2*dim+1:2*dim+4)) *pattern(observation_idx,:)')' + x(1:dim)', [], 1 );
        %H = @(x, observation_idx) reshape( (quat_to_mat(x(2*dim+1:2*dim+4)) *pattern(observation_idx,:)')' + x(1:dim)', [], 1 );
        %H = @(x, observation_idx) reshape( (Rot(x(2*dim+1:2*dim+4)) * pattern(observation_idx,:)')' + x(1:dim)', [], 1 );
        
        %J = precomputeJacobian(JRow1, JRow2, JRow3);
        
        s(1).H = H;
        s(1).J = J;
        % The measurment covariance matrix
        R = obs_noise*eye(observation_dim);
        s(1).R = R;
        
        global_params.R = R;
        global_params.H = H;
        global_params.pattern = pattern;
        
        % Do not specify an initial state
        s(1).x = nan;
        s(1).P = nan;
end
    function J = precomputeJacobian(JRow1, JRow2, JRow3)
        % For speed avoid too many functions, just use 1 ugly one
        J = @(x) [repmat(eye(3), 4, 1) zeros(12,3) ...
                     [pattern(1,:) * JRow1(x(2*dim+1), x(2*dim+2), x(2*dim+3), x(2*dim+4));
                      pattern(1,:) * JRow2(x(2*dim+1), x(2*dim+2), x(2*dim+3), x(2*dim+4));
                      pattern(1,:) * JRow3(x(2*dim+1), x(2*dim+2), x(2*dim+3), x(2*dim+4));

                      pattern(2,:) * JRow1(x(2*dim+1), x(2*dim+2), x(2*dim+3), x(2*dim+4));
                      pattern(2,:) * JRow2(x(2*dim+1), x(2*dim+2), x(2*dim+3), x(2*dim+4));
                      pattern(2,:) * JRow3(x(2*dim+1), x(2*dim+2), x(2*dim+3), x(2*dim+4));

                      pattern(3,:) * JRow1(x(2*dim+1), x(2*dim+2), x(2*dim+3), x(2*dim+4));
                      pattern(3,:) * JRow2(x(2*dim+1), x(2*dim+2), x(2*dim+3), x(2*dim+4));
                      pattern(3,:) * JRow3(x(2*dim+1), x(2*dim+2), x(2*dim+3), x(2*dim+4));

                      pattern(4,:) * JRow1(x(2*dim+1), x(2*dim+2), x(2*dim+3), x(2*dim+4));
                      pattern(4,:) * JRow2(x(2*dim+1), x(2*dim+2), x(2*dim+3), x(2*dim+4));
                      pattern(4,:) * JRow3(x(2*dim+1), x(2*dim+2), x(2*dim+3), x(2*dim+4));] ...
                  zeros(12,4) ];
    end
end


