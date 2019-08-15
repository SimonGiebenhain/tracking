function s = correctKalman(s, noKnowledge, globalParams, missedDetections)
%CORRECTKALMAN Summary of this function goes here
%   Detailed explanation goes here

R = globalParams.R;
pattern = globalParams.pattern;

nMarkers = size(pattern,1);
dim = size(pattern,2);

% If we don't know which detections are missing, we need to come up
% with a prediction for what detections are missing (the detections of which marker that is), i.e. we need to
% find H and R which best explain the measurements.
if noKnowledge 
    detections = reshape(s.z, [], dim);
    % Find an assignment between the individual markers and the detections
    
    if size(detections,1) > 2
        assignment = match_patterns(pattern, detections, 'new', s);
        assignment = assignment(1,1:size(detections,1));
    else
        assignment = match_patterns(pattern, detections - s.x(1:dim)', 'ML', s);
    end
    
    % Print in case an error was made in the assignment
    %inversions = assignment(2:end) - assignment(1:end-1);
    %if min(inversions) < 1
    %    assignment
    %else
    %    fprintf('yay\n')
    %end
    
    % construct H and R from assignment vector, i.e. delete the
    % corresponding rows in H and R.
    detectionsIdx = zeros(dim*length(assignment),1);
    for i = 1:dim
        detectionsIdx( (i-1)*length(assignment) + 1: (i-1)*length(assignment) + length(assignment)) = assignment' + nMarkers*(i-1);
    end
    %s.H = @(x) s.H(x,assignment);
    s.R = R(detectionsIdx, detectionsIdx);
else
    detectionsIdx = ~missedDetections;
end


% Correction based on observation
J = s.J;
subindex = @(A, rows) A(rows, :);     % for row sub indexing
J = @(x) subindex(J(x(7), x(8), x(9), x(10)), detectionsIdx);

%Evaluate Jacobian at current position
J = J(s.x);
% Compute Kalman gain factor:
K = s.P*J'/(J*s.P*J'+s.R);

z = s.H(s.x);
s.x = s.x + K*(s.z-z(detectionsIdx));

% TODO immer machen? oder nur wenn detection?
% Correct covariance matrix estimate
s.P = s.P - K*J*s.P;

end

