function s = kalman_step(s, no_knowledge, global_params, model, missed_detections, quat)
%KALMAN_STEP Summary of this function goes here
%   Detailed explanation goes here

R = global_params.R;
pattern = global_params.pattern;
H = global_params.H;

num_markers = size(pattern,1);
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
                assignment = match_patterns(pattern, reshape(detections, [],dim), 'ML');
                
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
            if isfield(s, 'z') && sum(isnan(s.z)) < 1
                detections = reshape(s.z, [], dim);
                assignment = match_patterns(pattern, detections, 'edges');
                position_estimate = mean(detections - pattern(assignment,:),1);
                s.x = [position_estimate'; zeros(dim,1); 0.25*ones(4,1); zeros(4,1)];
                s.P = eye(2*dim+8) .* repelem([global_params.init_pos_var; global_params.init_motion_var; global_params.init_quat_var; global_params.init_quat_mot_var], [dim, dim, 4, 4]);
            else
                % If we don't even have an observation, initialize at a random
                % position and zero velocity
                s.x = [rand(dim,1); zeros(3,1); 0.25*ones(4,1); zeros(4,1)];
                s.P = eye(2*dim+8) .* repelem([global_params.init_motion_var],2*dim+8)';
            end
            
        else
            % If we don't know which detections are missing, we need to come up
            % with a prediction for what detections are missing (the detections of which marker that is), i.e. we need to
            % find H and R which best explain the measurements.
            if no_knowledge &&  isfield(s,'z') && sum(isnan(s.z)) < 1
                detections = s.z;
                % Find an assignment between the individual markers and the detections
                Rot = quat_to_mat();
                assignment = match_patterns(pattern, reshape(detections, [],dim) - s.x(1:dim)', 'ML', Rot(s.x(2*dim+1:2*dim+4)));
                
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
                %s.H = @(x) s.H(x,assignment);
                s.R = R(detections_idx, detections_idx);
            end
            
            % partial derivatives of rotation matrix w.r.t. quaternion
            % components
            %             partial_q1 = @(q) 2 * [0 -q(4) q(3); q(4) 0 -q(2); -q(3) q(2) 0          ] * 1/sum(q.^2) - quat_to_mat(q,0)* 1/sum(q.^2)^2 * 2*q(1);
            %             partial_q2 = @(q) 2 * [0 q(3) q(4); q(3) -2*q(2) -q(1); q(4) q(1) -2*q(2)] * 1/sum(q.^2) - quat_to_mat(q,0)* 1/sum(q.^2)^2 * 2*q(2);
            %             partial_q3 = @(q) 2 * [-2*q(3) q(2) q(1); q(2) 0 q(4); -q(1) q(4) -2*q(3)] * 1/sum(q.^2) - quat_to_mat(q,0)* 1/sum(q.^2)^2 * 2*q(3);
            %             partial_q4 = @(q) 2 * [-2*q(4) -q(1) q(2); q(1) -2*q(4) q(3); q(2) q(3) 0] * 1/sum(q.^2) - quat_to_mat(q,0)* 1/sum(q.^2)^2 * 2*q(4);
            
            % Supply quat as known, doesn't really make sense, doesn't work
            %s.x(2*dim+1:2*dim+4) = quat;
            %q = quat;
            
            % Calculate Jacobian
            %             q = s.x(2*dim+1:2*dim+4);
            %             part_of_J = @(pat,q) [eye(3) zeros(3) partial_q1(q)*pat partial_q2(q)*pat partial_q3(q)*pat partial_q4(q)*pat zeros(3,4)];
            %             J = zeros(num_markers*dim, 2*dim + 8);
            %             for i = 0:num_markers-1
            %                 J(i*dim+1:(i+1)*dim,:) = part_of_J(pattern(i+1,:)',q);
            %             end
            
            
            % Here the actal extended kalman filter begins:
            
            % Prediction for state vector and covariance:
            s.x = s.A*s.x;
            s.P = s.A * s.P * s.A' + s.Q;
            
            
            
            % Correction based on observation (if observation is present):
            if isfield(s,'z') && sum(isnan(s.z)) < 1
                
                J = s.J;
                subindex = @(A, rows) A(rows, :);     % for row sub indexing
                if no_knowledge
                    J = @(x) subindex(J(x(7), x(8), x(9), x(10)), detections_idx);
                else
                    J = @(x) subindex(J(x(7), x(8), x(9), x(10)), ~missed_detections);
                end
                
                %Evaluate Jacobian at current position
                J = J(s.x);
                % Compute Kalman gain factor:
                K = s.P*J'*inv(J*s.P*J'+s.R);
                
                if no_knowledge
                    z = s.H(s.x);
                    %s.x = s.x + K*(s.z-z(repmat(assignment',3,1)));
                    s.x = s.x + K*(s.z-z(detections_idx));

                else
                    %Rot = @(q) subs(global_params.H.rot, [q1; q2; q3; q4], q );
                    %H = @(x) reshape( (Rot(x(2*dim+1:2*dim+4)) *pattern(~missed_detections(1:3:end),:)')' + x(1:dim)', [], 1 );
                    z = s.H(s.x);
                    s.x = s.x + K*(s.z-z(~missed_detections));
                end
                % TODO immer machen? oder nur wenn detection?
                % Correct covariance matrix estimate
                s.P = s.P - K*J*s.P;
            end
            
        end
        
end

