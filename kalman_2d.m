% credit for inital code structure:
% https://de.mathworks.com/matlabcentral/fileexchange/24486-kalman-filter-in-matlab-tutorial

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
% z = Hx + v       meaning the observation vector z is a linear function
%                  of the state vector, and this linear relationship is
%                  represented by premultiplication by "observation
%                  matrix" H. There is also gaussian measurement
%                  noise v.
% where w ~ N(0,Q) meaning w is gaussian noise with covariance Q
%       v ~ N(0,R) meaning v is gaussian noise with covariance R
%
% VECTOR VARIABLES:
%
% s.x = state vector estimate. In the input struct, this is the
%       "a priori" state estimate (prior to the addition of the
%       information from the new observation). In the output struct,
%       this is the "a posteriori" state estimate (after the new
%       measurement information is included).
% s.z = observation vector
% s.u = input control vector, optional (defaults to zero).
%
% MATRIX VARIABLES:
%
% s.A = state transition matrix (defaults to identity).
% s.P = covariance of the state vector estimate. In the input struct,
%       this is "a priori," and in the output it is "a posteriori."
%       (required unless autoinitializing as described below).
% s.B = input matrix, optional (defaults to zero).
% s.Q = process noise covariance (defaults to zero).
% s.R = measurement noise covariance (required).
% s.H = observation matrix (defaults to identity).
%
% NORMAL OPERATION:
%
% (1) define all state definition fields: A,B,H,Q,R
% (2) define intial state estimate: x,P
% (3) obtain observation and control vectors: z,u
% (4) call the filter to obtain updated state estimate: x,P
% (5) return to step (3) and repeat
%
% INITIALIZATION:
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

global pattern;
% pattern = [0.5; 1; -0.1; -2];
pattern = [1 1; 0 0; 0 -2; -2 0];

% global intra_pattern;
% intra_pattern = zeros(length(pattern),length(pattern));
%
% for i = 1:length(pattern)
%     marker = pattern(i);
%     for j = 1:length(pattern)
%         intra_pattern(i,j) = marker - pattern(j);
%     end
% end

% Define the system as a constant of 12 volts:
clear s
%s.x = 12;
s.A = [1 0 1 0 0; 0 1 0 1 0; 0 0 1 0 0; 0 0 0 1 0; 0 0 0 0 1];
% Define a process noise (stdev) of 2 volts as the car operates:
s.Q = 10*eye(5); % variance, hence stdev^2
s.Q(5,5) = 0;
% Define the voltimeter to measure the voltage itself:
global H;
H = [ [ones(4,1); zeros(4,1)] [zeros(4,1); ones(4,1)] zeros(8,1) zeros(8,1) pattern(:)];
s.H = H;
% Define a measurement error (stdev) of 2 volts:
global R;
R = 40^2*eye(8); % variance, hence stdev^2
s.R = R;
% Do not define any system input (control) functions:
s.B = 0;
s.u = 0;
% Do not specify an initial state:
s.x = nan;
s.P = nan;
% Generate random voltages and watch the filter operate.
tru=[]; % truth voltage

no_knowledge = 1;

T = 1000;
for t=1:2:T
    tru(end+1,:) = [sqrt(t/2)*1/10*sin(t/32) sin(t/30)*cos(t/68)^2*5];
    if rand > 0.15    
    missed_detections = rand(4,1) < 0.3;
    missed_detections = [missed_detections; missed_detections];
    if sum(~missed_detections) < 4
        in = randi(4);
        missed_detections(in) = 0;
        missed_detections(in+4) = 0;
        missed_detections( mod(in,4)+1 ) = 0;
        missed_detections( mod(in,4)+1+4 ) = 0;
        
    end
    s(end).H = H(~missed_detections,:);
    s(end).R = R(~missed_detections, ~missed_detections);
    s(end).z = s(end).H * [tru(end,:)'; 0; 0; 1] + mvnrnd(zeros(size(s(end).H,1),1), 0.005*eye(size(s(end).H,1)),1)'; % create a measurement
    if no_knowledge
        s(end).H = H;
        s(end).R = R;
    end
    
    
    else
        s(end).z = [NaN;NaN;NaN;NaN;NaN;NaN;NaN;NaN];
    end
    s(end+1)=kalmanf(s(end), no_knowledge); % perform a Kalman filter iteration
end

%%
figure
hold on
grid on

scatter([-5,5], [-5,5])
% plot measurement data:
detections = reshape(s(1).z,[],2);
markers = scatter(detections(:,1), detections(:,2) , 'r.');

positions = [s(2:end).x]';
%states = scatter(s(2).x(1), s(2).x(2), 'b*');
plot(positions(2:3,1), positions(2:3,2), 'b-')


%truths = scatter(tru(1,1), tru(1,2) ,'g');
plot(tru(1:2,1),tru(1:2,2), 'g-')

for t=2:T
    detections = reshape(s(t).z,[],2);
    
    markers.XData = detections(:,1);
    markers.YData = detections(:,2);
    plot(positions(t:t+1,1), positions(t:t+1,2), 'b-')
    plot(tru(t:t+1,1),tru(t:t+1,2), 'g-')
    
    %states.XData = s(t+1).x(1);
    %states.YData = s(t+1).x(2);
    
    %truths.XData = tru(t,1);
    %truths.YData = tru(t,2);
    
    
    drawnow
    pause(0.01)
    
end


%legend([hz hk ht],'observations','Kalman output','true voltage')
title('Automobile Voltimeter Example')
hold off
%figure;
%plot(states(:,2)); hold on;
%plot(states(:,3)); hold off;

function s = kalmanf(s, no_knowledge)
global pattern
global R;
global H;


% set defaults for absent fields:
if ~isfield(s,'x'); s.x=nan*z; end
if ~isfield(s,'P'); s.P=nan; end
%if ~isfield(s,'z'); error('Observation vector missing'); end
if ~isfield(s,'u'); s.u=0; end
if ~isfield(s,'A'); s.A=eye(length(x)); end
if ~isfield(s,'B'); s.B=0; end
if ~isfield(s,'Q'); s.Q=zeros(length(x)); end
if ~isfield(s,'R'); error('Observation covariance missing'); end
if ~isfield(s,'H'); s.H=eye(length(x)); end

if isnan(s.x)
    % initialize state estimate from first observation
    %if diff(size(s.H))
    %error('Observation matrix must be square and invertible for state autointialization.');
    %end
    %s.x = inv(s.H)*s.z;
    
    %TODO
    
    if isfield(s, 'z') && sum(isnan(s.z)) < 1
        n = size(s.z,1);
        s.x = [mean(s.z(1:n/2)); mean(s.z(n/2+1:n)); 0.25; 0.25; 1];
        s.P = [30 0 0 0 0; 0 30 0 0 0; 0 0 1000 0 0; 0 0 0 1000 0; 0 0 0 0 0];
    else
        s.x = [rand; rand; 0; 0; 1];
        s.P = [100 0 0 0 0; 0 100 0 0 0; 0 0 1000 0 0; 0 0 0 1000 0; 0 0 0 0 0];
    end
    
    
    %s.P = inv(s.H)*s.R*inv(s.H');
else
    % If we don't know which detections are missing, we need to come up
    % with a prediction for what detections are missing, i.e. we need to
    % find H and R which best explain the measurements.
    if no_knowledge &&  isfield(s,'z') && sum(isnan(s.z)) < 1
        detections = s.z;
        num_detections = size(detections,1);
        diff = zeros(num_detections);
        
        assignment = match_patterns(pattern, reshape(detections, [],2));
        inversions = assignment(2:end) - assignment(1:end-1);
        if min(inversions) < 1
            assignment
        end
        % construct H from assignment vector
        s.H = H([assignment';assignment'+4],:);
        s.R = R([assignment';assignment'+4], [assignment';assignment'+4]);
    end
    % This is the code which implements the discrete Kalman filter:
    
    % Prediction for state vector and covariance:
    s.x = s.A*s.x;
    s.P = s.A * s.P * s.A' + s.Q;
    % Compute Kalman gain factor:
    K = s.P*s.H'*inv(s.H*s.P*s.H'+s.R);
    
    % Correction based on observation (if observation is present):
    if isfield(s,'z') && sum(isnan(s.z)) < 1
        s.x = s.x + K*(s.z-s.H*s.x);
    end
    s.P = s.P - K*s.H*s.P;
    
    % Note that the desired result, which is an improved estimate
    % of the sytem state vector x and its covariance P, was obtained
    % in only five lines of code, once the system was defined. (That's
    % how simple the discrete Kalman filter is to use.) Later,
    % we'll discuss how to deal with nonlinear systems.
    
end

end



