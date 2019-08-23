function s = initializeKalman(s, globalParams)
%INITIALIZEKALMAN Summary of this function goes here
%   Detailed explanation goes here


% initialize state estimate from first observation
%if diff(size(s.H))
%error('Observation matrix must be square and invertible for state autointialization.');
%end
%s.x = inv(s.H)*s.z;
%s.P = inv(s.H)*s.R*inv(s.H');


pattern = globalParams.pattern;
dim = size(pattern,2);

% TODO better initial guess
% TODO how to initially guess the velocity?
detections = reshape(s.z, [], dim);
assignment = match_patterns(pattern, detections, 'ML', Rot(0.25*ones(4,1)));
positionEstimate = mean(detections - pattern(assignment,:),1);
s.x = [positionEstimate'; zeros(dim,1); 0.25*ones(4,1); zeros(4,1)];
s.P = eye(2*dim+8) .* repelem([globalParams.initPositionVar; globalParams.initMotionVar; globalParams.initQuatVar; globalParams.initQuatMotionVar], [dim, dim, 4, 4]);


