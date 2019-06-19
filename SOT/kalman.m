%{
% My system currently tracks a single object in 3d space. Observations are
% made not directly of the position of the object, but of 4 markers which
% are arranged in a predefined pattern.
%
% TODO: So far the orientation of the pattern is fixed.
%
% The system evolves according to the following difference equations,
% where quantities are further defined below:
%
% x = Ax + Bu + w  meaning the state vector x evolves during one time
%                  step by premultiplying by the "state transition
%                  matrix" A. There is optionally (if nonzero) an input
%                  vector u which affects the state linearly, and this
%                  linear effect on the state is represented by
%                  premultiplying by the "input matrix" B. There is also
%                  gaussian process noise w.
%                  There is no B and u in my model.
% 
% z = Hx + v       meaning the observation vector z is a linear function
%                  of the state vector, and this linear relationship is
%                  represented by premultiplication by "observation
%                  matrix" H. There is also gaussian measurement
%                  noise v.
%                  In my concrete example H maps from the 3d position to
%                  the positions of the 4 markers.
%
% where w ~ N(0,Q) meaning w is gaussian noise with covariance Q
%       v ~ N(0,R) meaning v is gaussian noise with covariance R
%
%
% VECTOR VARIABLES:
%
% s.x = state vector estimate. 
%       In my code this is 7 dimensional. 3 dimensions are used to store
%       the position. 3 dimensions are used to store the current velocity
%       in 3d space and 1 dimension is fixed to 1. This is necessary to
%       incorporate the pattern of markers which are observed.
% 
% s.z = observation vector.
%       In my code this is 12 dimensional, as there are 4 markers and each
%       has a 3d position
%
%
% MATRIX VARIABLES:
%
% s.A = state transition matrix.
%       In my code this maps the old position to the old position plus the
%       velocity as an offset.

% s.P = covariance of the state vector estimate.

% s.Q = process noise covariance matrix.
   
% s.R = measurement noise covariance matrix.

% s.H = observation matrix.
%       In my code this is a 12 x 7 matrix and maps the state x to an
%       observation z. The marker positions are calculated by adding an
%       offset to the position of the object.
%
%
% TODO: INITIALIZATION:
%
% If an initial state estimate is unavailable, it can be obtained
% from the first observation as follows, provided that there are the
% same number of observable variables as state variables. This "auto-
% intitialization" is done automatically if s.x is absent or NaN.
%
% x = inv(H)*z
% P = inv(H)*R*inv(H')
%
% This is mathematically equivalent to setting the initial state estimate
% covariance to infinity.
%
%
% Credit for inital code structure:
% https://de.mathworks.com/matlabcentral/fileexchange/24486-kalman-filter-in-matlab-tutorial
%}
clear s

% The pattern of markers
pattern = [1 1 1; 0 0 0; 1 0 -2; -0.5 -2 0.5];

% The number of timestep I run the simulation
T = 1000;
[s, global_params] = setup_kalman(pattern, T, 'linear', 2500, 5);

% Preallocate ground truth trajectory
tru=zeros(T,3);

% When no_knowledge is true the system doesn't know which detection
% corresponds to which marker.
no_knowledge = 1;

frame_drop_rate = 0.2;
marker_drop_rate = 0.4;
marker_noise = 0.01;

% Run the simulatio 
for t=1:T
    % Get the true trajectory from some crazy function
    %tru(end+1,:) = [sqrt(t/2)*1/10*sin(t/32) sin(t/30)*cos(t/68)^2*5, t/T*3];
    tru(t,:) = [t/T*5*sin(t/32) cos(t/68)^2*5, sin(t/20)^2*t/T*3];

    % In some frames drop all detections.
    if rand > frame_drop_rate
        % In some frames drop some individual markers
        missed_detections = rand(4,1) < marker_drop_rate;
        missed_detections = repmat(missed_detections, 3,1);
        
        % TODO handle cases when only 1 marker was detected per in a frame
        % For now: Make sure there are at least two markers detected
        if sum(~missed_detections) < 6
            in = randi(4);
            missed_detections(in) = 0;
            missed_detections(in+4) = 0;
            missed_detections(in+8) = 0;
            
            missed_detections( mod(in,4)+1 ) = 0;
            missed_detections( mod(in,4)+1+4 ) = 0;
            missed_detections( mod(in,4)+1+8 ) = 0;
        end
        
        % Delete some rows in H and R to accomodate for the missed
        % measurements.
        Hcur = H(~missed_detections,:);
        Rcur = R(~missed_detections, ~missed_detections);
        % Create the measurement
        s(t).z = Hcur * [tru(t,:)'; 0; 0; 0; 1] + mvnrnd(zeros(size(Hcur,1),1), marker_noise*eye(size(Hcur,1)),1)';
        
        % Decide whether the system knows which of which markers the
        % detections were dropped.
        if no_knowledge
            s(t).H = H;
            s(t).R = R;
        else
            s(t).H = Hcur;
            s(t).R = Rcur;
        end
        
    else
        % dropped frames don't have any detections
        s(t).z = [NaN;NaN;NaN;NaN;NaN;NaN;NaN;NaN;NaN;NaN;NaN;NaN];
    end
    s(t+1) = kalman_step(s(t), no_knowledge, global_params); % perform a Kalman filter iteration
end

%% Plot the results

figure;
hold on
grid on

% Initialize the range of display
scatter3([-6,6], [-6,6], [-5,5])

% plot detections in first frame
detections = reshape(s(1).z,[],3);
markers = scatter3(detections(:,1), detections(:,2), detections(:,3) , 'r.');

% Plot the positions (part of the state)
states = [s(2:end).x]';
plot3(states(2:3,1), states(2:3,2), states(2:3,3), 'b-')

% Plot the ground truth
plot3(tru(1:2,1),tru(1:2,2), tru(1:2,2), 'g-')

% update plot step by step
for t=2:T
    detections = reshape(s(t).z,[],3);
    
    % plot detections of next frame
    markers.XData = detections(:,1);
    markers.YData = detections(:,2);
    markers.ZData = detections(:,3);
    
    % add to the estimated and fround truth trajectory
    plot3(states(t:t+1,1), states(t:t+1,2), states(t:t+1,3), 'b-')
    plot3(tru(t:t+1,1), tru(t:t+1,2), tru(t:t+1,3), 'g-')
    drawnow
    pause(0.01)
end

hold off

%% Plot the velocities

figure;
plot(states(:,4), 'DisplayName', 'velocity in x direction'); hold on;
plot(states(:,5), 'DisplayName', 'velocity in y direction'); 
plot(states(:,6), 'DisplayName', 'velocity in z direction'); 
legend; hold off;