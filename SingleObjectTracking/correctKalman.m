function [s, rejectedDetections, c_ret] = correctKalman(s, globalParams, hyperParams, age, motionType)
%CORRECTKALMAN Summary of this function goes here
%   Detailed explanation goes here

c_ret = NaN;
R = globalParams.R;
pattern = globalParams.pattern;

nMarkers = size(pattern,1);
dim = size(pattern,2);

rejectedDetections = zeros(0, 3);

% If we don't know which detections are missing, we need to come up
% with a prediction for what detections are missing (the detections of which marker that is), i.e. we need to
% find H and R which best explain the measurements.
%if noKnowledge
skipFPFiltering = 0;
if strcmp(s.type, 'LG-EKF')
    detections = reshape(s.z, dim, [])';
    nDets = size(detections, 1);
    expectedPositions = measFuncNonHomogenous(s.mu, s.pattern);
    [dists, asg] = min(pdist2(detections, expectedPositions), [], 2);
    ds = detections - s.mu.X(1:3, 4)';
    if s.mu.motionModel == 0
        vNorm = 0;
    elseif s.mu.motionModel == 1
        vNorm = sqrt(sum((s.mu.v).^2));
    elseif s.mu.motionModel == 2
        vNorm = sqrt(sum((s.mu.v).^2));
    else
        'correctKalman: no motion type'
    end
    
else
    detections = reshape(s.z, [], dim);
    nDets = size(detections, 1);
    expectedPositions = measFuncNonHomogenous(s.mu, s.pattern);
    dists = min(pdist2(detections, expectedPositions), [], 2);
    ds = detections - s.x(1:dim)';
    vNorm = sqrt(sum(s.x(dim+1:2*dim).^2));    
end
rejectedDetectionsIdx = zeros(nDets, 1);
if age > 5 && s.mu.motionModel == 0 && hyperParams.doFPFiltering == 1 && ~skipFPFiltering
    threshold = hyperParams.minAssignmentThreshold + (vNorm/3)^2;% + (aNorm)^2;
    if nnz(dists > threshold) < nDets || s.consecutiveInvisibleCount > 10 %|| (s.flying < 1 && s.consecutiveInvisibleCount == 0 && nnz(dists > threshold) == 1)
        rejectedDetectionsIdx = dists > threshold;
        rejectedDetections = detections(rejectedDetectionsIdx, :);
        ds(rejectedDetectionsIdx, :) = [];
        detections(rejectedDetectionsIdx, :) = [];
    end
end

deltas = s.latest5pos - [s.latest5pos(end, :); s.latest5pos(1:end-1, :)];
deltas(s.latestPosIdx+1, :) = [];
delta = norm(sum(deltas, 1));


if s.mu.motionModel == 0
    if mean(dists) > 20 && ( delta > 45 && s.mu.motionModel == 0 && s.framesInNewMotionModel > 10 || age < 3)
            vEst = norm( mean(detections) - mean(expectedPositions) );
            s = switchMotionModel(s, vEst, 2, hyperParams);
    end
elseif s.mu.motionModel == 1
    s.mu.motionModel
elseif s.mu.motionModel == 2
    if s.mu.motionModel == 2 && norm(s.mu.v) < 4.0 && s.framesInNewMotionModel > 1
            s = switchMotionModel(s, -1, 0, hyperParams);
    end
end



if size(ds, 1) >= 1
    s.consecutiveInvisibleCount = 0;
    if size(ds, 1) <= 1
        if strcmp(s.type, 'LG-EKF')
            [p, ~, certainty, method] = match_patterns(pattern, ds, 'ML', s.mu.X(1:3, 1:3), hyperParams);
        else
            if strcmp(motionType, 'constAcc')
                quatIdx = 3*dim+1:3*dim+4;
            else
                quatIdx = 2*dim+1:2*dim+4;
            end
            [p, ~, certainty, method] = match_patterns(pattern, ds, 'ML', Rot(s.x(quatIdx)), hyperParams);
            s.z = reshape(detections, [], 1);
        end
    else
        if strcmp(s.type, 'LG-EKF')
            % Check if predicted marker positions are good fit already
            % if so don't do costly pattern matching algorithm
            asg = asg(rejectedDetectionsIdx == 0);
            mse = mean(sqrt(sum(( detections - expectedPositions(asg, :) ).^2, 2)));
            if mse < 2 && nDets >= 3 && length(asg) == length(unique(asg))
                lostDs = zeros(0,1);
                certainty = 5*mse + (4-length(asg))*hyperParams.costOfNonAsDtMA;
                p = asg';
                method = 'final4';
            else
                [p, lostDs, certainty, method, c_ret] = match_patterns(pattern, ds, 'final4', s.mu.X(1:3,1:3), hyperParams);
            end
        else
            if strcmp(motionType, 'constAcc')
                quatIdx = 3*dim+1:3*dim+4;
            else
                quatIdx = 2*dim+1:2*dim+4;
            end
            [p, lostDs, certainty, method] = match_patterns(pattern, ds, 'final4', Rot(s.x(quatIdx)), hyperParams);
            s.z = reshape(detections, [], 1);
            
        end
    end

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
            if size(assignment, 2) >= 2
                divisor = size(assignment, 2)^3+1;
            else
                divisor = 3;
            end
            certainty = (certainty/hyperParams.certaintyScale)^2 / divisor;
            certainty = max(0.05, certainty);%0.05
            certainty = min(150, certainty);%150
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
        if s.consecutiveInvisibleCount > 2
            s.R = s.R/1.5;
        end
    else
        s.R = R(detectionsIdx, detectionsIdx);
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
            HLinSE3AC(m(1,1), m(1,2), m(1,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3), s.mu.motionModel);
            HLinSE3AC(m(2,1), m(2,2), m(2,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3), s.mu.motionModel);
            HLinSE3AC(m(3,1), m(3,2), m(3,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3), s.mu.motionModel);
            HLinSE3AC(m(4,1), m(4,2), m(4,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3), s.mu.motionModel)
            ];
        
        H = @(S) jac(s.pattern, S.X);
        H = H(s.mu);
        H = H(detectionsIdx, :);
        K = s.P*H'/(H*s.P*H'+s.R);
        predErr = z(~lostIdx)-zPred(detectionsIdx);
        if length(predErr) >= 12
            c_ret = mean(predErr.^2);
        end
        m = K*predErr;
        s.mu = comp(s.mu, expSE3ACvec(m));
        if s.mu.motionModel == 0
            s.P = (eye(6)- K*H)*s.P;
            
        elseif s.mu.motionModel == 1
            s.P = (eye(9)- K*H)*s.P;
            
        elseif s.mu.motionModel == 2
            s.P = (eye(12)- K*H)*s.P;
            
        else
            'correctkalman: unexpecte motion model'
        end
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
else
    s.consecutiveInvisibleCount = s.consecutiveInvisibleCount + 1;
end


end