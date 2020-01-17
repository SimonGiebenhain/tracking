function [s, assignment] = correctKalman(s, noKnowledge, globalParams, missedDetections, hyperParams, age)
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
    nDets = size(detections, 1);
    % Find an assignment between the individual markers and the detections
    ds = detections - s.x(1:dim)';
    if age > 3 && hyperParams.doFPFiltering == 1
        norms = sqrt(sum(ds.^2, 2));
        threshold = hyperParams.whenFPFilter + sqrt(sum(s.x(dim+1:2*dim).^2))*4;
        if nnz(norms > threshold) < nDets || nnz(norms > threshold) == 1
            ds(norms > threshold, :) = [];
            detections(norms > threshold, :) = [];
        end
    end

    if size(ds, 1) <= 1
        [p, ~, FPs, certainty, method] = match_patterns(pattern, ds, 'ML', s, hyperParams);
    else
        [p, lostDs, FPs, certainty, method] = match_patterns(pattern, ds, 'final2', s, hyperParams);
    end
    %[p, ~,  FPs, certainty, method] = match_patterns(pattern, detections - s.x(1:dim)', 'correct', s, hyperParams);
    %detections = detections(~FPs, :);
    s.z = reshape(detections, [], 1);
    if strcmp(method, 'new')
        assignment(p) = 1:length(p);
        assignment = assignment(1:size(detections,1));
        lostDet = assignment == 5;
        assignment = assignment(assignment ~= 5);
        if hyperParams.useAssignmentLength == 1
            certainty = certainty / (size(assignment, 2) + 1);
        end
    elseif strcmp(method, 'ML')
        assignment = p;
        lostDet = zeros(size(assignment));
        if hyperParams.useAssignmentLength == 1
            certainty = hyperParams.certaintyFactor * certainty / (size(assignment, 2) + 1);
        end
    elseif strcmp(method, 'final2')
        nDets = size(detections, 1);
        assignment = p;
        lostDet = zeros(nDets, 1);
        lostDet(lostDs) = 1;
        if hyperParams.useAssignmentLength == 1
            certainty = (certainty/hyperParams.certaintyScale)^2 / (size(assignment, 2) + 1);
        end
    else 
        fprintf('unknown method encountered')
    end
    
    % assignment holds the marker index assigned to the detections
    
    % construct H and R from assignment vector, i.e. delete the
    % corresponding rows in H and R.
    detectionsIdx = zeros(dim*length(assignment),1);
    lostIdx = zeros(dim*length(lostDet),1);
    for i = 1:dim
        detectionsIdx( (i-1)*length(assignment) + 1: i*length(assignment)) = assignment' + nMarkers*(i-1);
        lostIdx((i-1)*length(lostDet) + 1: i*length(lostDet)) = lostDet';
    end
    if hyperParams.adaptiveNoise == 1
        s.R = certainty * R(detectionsIdx, detectionsIdx);
    else
        s.R = R(detectionsIdx, detectionsIdx);
    end
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
s.x = s.x + K*(s.z(~lostIdx)-z(detectionsIdx));

% TODO immer machen? oder nur wenn detection?
% Correct covariance matrix estimate
s.P = s.P - K*J*s.P;

end
