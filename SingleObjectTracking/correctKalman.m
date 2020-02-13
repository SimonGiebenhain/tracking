function [s, assignment] = correctKalman(s, noKnowledge, globalParams, missedDetections, hyperParams, age, motionType)
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
    if strcmp(s.type, 'LG-EKF')
        detections = reshape(s.z, dim, [])';
        nDets = size(detections, 1);
        ds = detections - s.mu.X(1:3, 4)';
        vNorm = sqrt(sum((s.mu.v).^2));
    else
        detections = reshape(s.z, [], dim);
        nDets = size(detections, 1);
        ds = detections - s.x(1:dim)';
        vNorm = sqrt(sum(s.x(dim+1:2*dim).^2));
    end
    if age > 10 && hyperParams.doFPFiltering == 1
        norms = sqrt(sum(ds.^2, 2));
        threshold = hyperParams.minAssignmentThreshold + vNorm*5;
        if nnz(norms > threshold) < nDets || nnz(norms > threshold) == 1
            ds(norms > threshold, :) = [];
            detections(norms > threshold, :) = [];
        end
    end
    
    if size(ds, 1) <= 1
        if strcmp(s.type, 'LG-EKF')
            [p, ~, FPs, certainty, method] = match_patterns(pattern, ds, 'ML', s.mu.X(1:3, 1:3), hyperParams);
        else
            if strcmp(motionType, 'constAcc')
                quatIdx = 3*dim+1:3*dim+4;
            else
                quatIdx = 2*dim+1:2*dim+4;
            end
            [p, ~, FPs, certainty, method] = match_patterns(pattern, ds, 'ML', Rot(s.x(quatIdx)), hyperParams);
            s.z = reshape(detections, [], 1);
        end
    else
        if strcmp(s.type, 'LG-EKF')
            [p, lostDs, FPs, certainty, method] = match_patterns(pattern, ds, 'final4', s.mu.X(1:3,1:3), hyperParams);
        else
            if strcmp(motionType, 'constAcc')
                quatIdx = 3*dim+1:3*dim+4;
            else
                quatIdx = 2*dim+1:2*dim+4;
            end
            [p, lostDs, FPs, certainty, method] = match_patterns(pattern, ds, 'final4', Rot(s.x(quatIdx)), hyperParams);
            s.z = reshape(detections, [], 1);
            
        end
    end
    %[p, ~,  FPs, certainty, method] = match_patterns(pattern, detections - s.x(1:dim)', 'correct', s, hyperParams);
    %detections = detections(~FPs, :);
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
        
        if vNorm >= 25
            certainty = certainty / 2;
        elseif vNorm < 10
            certainty = 3*certainty;
        end
    elseif strcmp(method, 'final3')
        nDets = size(detections, 1);
        assignment = p;
        lostDet = zeros(nDets, 1);
        lostDet(lostDs) = 1;
        if hyperParams.useAssignmentLength == 1
            certainty = (certainty/hyperParams.certaintyScale)^2 / (size(assignment, 2) + 1);
        end
    elseif strcmp(method, 'final4')
        nDets = size(detections, 1);
        assignment = p;
        lostDet = zeros(nDets, 1);
        lostDet(lostDs) = 1;
        if hyperParams.useAssignmentLength == 1
            certainty = (certainty/hyperParams.certaintyScale)^3 / (size(assignment, 2) + 1);
            certainty = max(0.005, certainty);
            certainty = min(100, certainty);
        end
    else
        fprintf('unknown method encountered')
    end
    
    % assignment holds the marker index assigned to the detections
    
    % construct H and R from assignment vector, i.e. delete the
    % corresponding rows in H and R.
    if strcmp(s.type, 'LG-EKF')
        detectionsIdx = zeros((dim+1)*length(assignment),1);
        lostIdx = repelem(lostDet, 4, 1);
        for i=1:length(assignment)
            for j=1:dim+1
                detectionsIdx((i-1)*(dim+1)+j) = (assignment(i)-1)*(dim+1)+j;
            end
        end
    else
        detectionsIdx = zeros(dim*length(assignment),1);
        lostIdx = zeros(dim*length(lostDet),1);
        for i = 1:dim
            detectionsIdx( (i-1)*length(assignment) + 1: i*length(assignment)) = assignment' + nMarkers*(i-1);
            lostIdx((i-1)*length(lostDet) + 1: i*length(lostDet)) = lostDet';
        end
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
if strcmp(s.type, 'LG-EKF')
    z = [detections ones(size(detections,1),1)];
    z = reshape(z', [],1);
    
    h = @(S) [measFunc(S,[pattern(1,:)';1]);
        measFunc(S,[pattern(2,:)';1]);
        measFunc(S,[pattern(3,:)';1]);
        measFunc(S,[pattern(4,:)';1])];
    
    zPred = h(s.mu);
    
    jac = @(m, mu) [
        HLinSE3AC(m(1,1), m(1,2), m(1,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3));
        HLinSE3AC(m(2,1), m(2,2), m(2,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3));
        HLinSE3AC(m(3,1), m(3,2), m(3,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3));
        HLinSE3AC(m(4,1), m(4,2), m(4,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3))
        ];
    
    H = @(S) jac(s.pattern, S.X);
    H = H(s.mu);
    H = H(detectionsIdx, :);
    
    K = s.P*H'/(H*s.P*H'+s.R);
    m = K*(z(~lostIdx)-zPred(detectionsIdx));
    s.mu = comp(s.mu, expSE3ACvec(m));
    s.P = (eye(12)- K*H)*s.P;
else
    %Correction based on observation
    Jac = s.J;
    subindex = @(A, rows) A(rows, :);     % for row sub indexing
    if strcmp(motionType, 'constAcc')
        J = @(x) subindex(Jac(x(10), x(11), x(12), x(13)), detectionsIdx);
    else
        J = @(x) subindex(Jac(x(7), x(8), x(9), x(10)), detectionsIdx);
    end
    
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

end









% function [s, assignment] = correctKalman(s, noKnowledge, globalParams, missedDetections, hyperParams, age, motionType)
% %CORRECTKALMAN Summary of this function goes here
% %   Detailed explanation goes here
%
% R = globalParams.R;
% pattern = globalParams.pattern;
%
% nMarkers = size(pattern,1);
% dim = size(pattern,2);
%
% % If we don't know which detections are missing, we need to come up
% % with a prediction for what detections are missing (the detections of which marker that is), i.e. we need to
% % find H and R which best explain the measurements.
% if noKnowledge
%     % Find an assignment between the individual markers and the detections
%     if strcmp(s.type, 'LG-EKF')
%         detections = reshape(s.z, dim, [])';
%         nDets = size(detections, 1);
%         ds = detections - s.mu(1:3, 4)';
%         vNorm = sqrt(sum((s.v).^2));
%     else
%         detections = reshape(s.z, [], dim);
%         nDets = size(detections, 1);
%         ds = detections - s.x(1:dim)';
%         vNorm = sqrt(sum(s.x(dim+1:2*dim).^2));
%     end
%     if age > 10 && hyperParams.doFPFiltering == 1
%         norms = sqrt(sum(ds.^2, 2));
%         threshold = hyperParams.minAssignmentThreshold + vNorm*5;
%         if nnz(norms > threshold) < nDets || nnz(norms > threshold) == 1
%             ds(norms > threshold, :) = [];
%             detections(norms > threshold, :) = [];
%         end
%     end
%
%
%     if size(ds, 1) <= 1
%         if strcmp(s.type, 'LG-EKF')
%             [p, ~, FPs, certainty, method] = match_patterns(pattern, ds, 'ML', s.mu(1:3, 1:3), hyperParams);
%         else
%             if strcmp(motionType, 'constAcc')
%                 quatIdx = 3*dim+1:3*dim+4;
%             else
%                 quatIdx = 2*dim+1:2*dim+4;
%             end
%             [p, ~, FPs, certainty, method] = match_patterns(pattern, ds, 'ML', Rot(s.x(quatIdx)), hyperParams);
%         end
%     else
%         if strcmp(s.type, 'LG-EKF')
%             [p, lostDs, FPs, certainty, method] = match_patterns(pattern, ds, 'final4', s.mu(1:3,1:3), hyperParams);
%         else
%             if strcmp(motionType, 'constAcc')
%                 quatIdx = 3*dim+1:3*dim+4;
%             else
%                 quatIdx = 2*dim+1:2*dim+4;
%             end
%             [p, lostDs, FPs, certainty, method] = match_patterns(pattern, ds, 'final4', Rot(s.x(quatIdx)), hyperParams);
%
%         end
%     end
%
%
%     if strcmp(method, 'new')
%         assignment(p) = 1:length(p);
%         assignment = assignment(1:size(detections,1));
%         lostDet = assignment == 5;
%         assignment = assignment(assignment ~= 5);
%         if hyperParams.useAssignmentLength == 1
%             certainty = certainty / (size(assignment, 2) + 1);
%         end
%     elseif strcmp(method, 'ML')
%         assignment = p;
%         lostDet = zeros(size(assignment));
%         if hyperParams.useAssignmentLength == 1
%             certainty = hyperParams.certaintyFactor * certainty / (size(assignment, 2) + 1);
%         end
%     elseif strcmp(method, 'final4')
%         nDets = size(detections, 1);
%         assignment = p;
%         lostDet = zeros(nDets, 1);
%         lostDet(lostDs) = 1;
%         if hyperParams.useAssignmentLength == 1
%             certainty = (certainty/hyperParams.certaintyScale)^2 / (size(assignment, 2) + 1);
%         end
%     else
%         fprintf('unknown method encountered')
%     end
%
%     % assignment holds the marker index assigned to the detections
%
%     % construct H and R from assignment vector, i.e. delete the
%     % corresponding rows in H and R.
%     if strcmp(s.type, 'LG-EKF')
%         detectionsIdx = zeros((dim+1)*length(assignment),1)
%         lostIdx = repelem(lostDet, 4, 1);
%         for i=1:length(assignment)
%             for j=1:dim+1
%                 detectionsIdx((i-1)*(dim+1)+j) = (assignment(i)-1)*(dim+1)+j;
%             end
%         end
%     else
%         detectionsIdx = zeros(dim*length(assignment),1);
%         lostIdx = zeros(dim*length(lostDet),1);
%         for i = 1:dim
%             detectionsIdx( (i-1)*length(assignment) + 1: i*length(assignment)) = assignment' + nMarkers*(i-1);
%             lostIdx((i-1)*length(lostDet) + 1: i*length(lostDet)) = lostDet';
%         end
%     end
%     if hyperParams.adaptiveNoise == 1
%         s.R = certainty * R(detectionsIdx, detectionsIdx);
%     else
%         s.R = R(detectionsIdx, detectionsIdx);
%     end
% else
%     detectionsIdx = ~missedDetections;
% end
%
%
% % Correction based on observation
% if strcmp(s.type, 'LG-EKF')
%
%     z = [detections ones(size(detections,1),1)];
%     z = reshape(z', [],1);
%
%     h = @(S) [measFunc(S,pattern(1,:)');
%         measFunc(S,pattern(2,:)');
%         measFunc(S,pattern(3,:)');
%         measFunc(S,pattern(4,:)')];
%
%     zPred = h(s.mu);
%
%     jac = @(m, mu) [
%         HLinSE3AC(m(1,1), m(1,2), m(1,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3));
%         HLinSE3AC(m(2,1), m(2,2), m(2,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3));
%         HLinSE3AC(m(3,1), m(3,2), m(3,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3));
%         HLinSE3AC(m(4,1), m(4,2), m(4,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3))
%         ];
%
%     H = @(S) jac(s.pattern, S.X);
%     H = H(s.mu);
%     H = H(detectionsIdx, :);
%
%     K = s.P*H'/(H*s.P*H'+s.R);
%     m = K*(z(~lostIdx)-zPred(detectionsIdx));
%     s.mu = comp(s.mu, expSE3ACvec(m));
%     s.P = (eye(12)- K*H)*s.P;
% else
%     %Correction based on observation
%     J = s.J;
%     subindex = @(A, rows) A(rows, :);     % for row sub indexing
%     if strcmp(motionType, 'constAcc')
%         J = @(x) subindex(J(x(10), x(11), x(12), x(13)), detectionsIdx);
%     else
%         J = @(x) subindex(J(x(7), x(8), x(9), x(10)), detectionsIdx);
%     end
%
%     %Evaluate Jacobian at current position
%     J = J(s.x);
%     % Compute Kalman gain factor:
%     K = s.P*J'/(J*s.P*J'+s.R);
%
%     z = s.H(s.x);
%     s.x = s.x + K*(s.z(~lostIdx)-z(detectionsIdx));
%
%     % TODO immer machen? oder nur wenn detection?
%     % Correct covariance matrix estimate
%     s.P = s.P - K*J*s.P;
% end
%
% end
