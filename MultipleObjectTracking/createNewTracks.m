function [tracks, ghostTracks, unassignedPatterns] = createNewTracks(detections, unassignedPatterns, tracks, patterns, params, patternNames, ghostTracks)
%CREATENEWTRACKS Summary of this function goes here
%   Detailed explanation goes here

if exist('ghostTracks','var') ~= 1
    kF = constructGhostKF([0 0 0], params);
    ghostTrack = struct(...
        'kalmanFilter', kF, ...
        'age', 1, ...
        'totalVisibleCount', 1, ...
        'consecutiveInvisibleCount', 0);
    ghostTracks(1) = ghostTrack;
    ghostTracks(:, 1) = [];
end

minDistToBird = 90;



% birds which were invisible for the last 5 frames or more are allowed to
% be re-initialized
invisCounts = [tracks(:).consecutiveInvisibleCount];
invis = invisCounts >= 5;
unassignedPatterns = unassignedPatterns | invis';

%TODO:
% allow to initialize ghostBirds even if all Patterns are already assigned?
if size(detections, 1) > 1 && sum(unassignedPatterns) > 0
    
    minClusterSize = 4;
    costOfNonAssignment = 0.7;
    
    
    %if sum(unassignedPatterns) == 1
    %minClusterSize = 3;
    %costOfNonAssignment = 0.5;
    %else
    %   minClusterSize = 4;
    %   costOfNonAssignment = 1;
    %end
    
    
    dim = size(patterns,3);
    epsilon = 55;
    clustersRaw = clusterUnassignedDetections(detections, epsilon);
    nClusters = 0;
    nPotentialGhosts = 0;
    potentialBirds = {};
    potentialGhosts = {};
    for i=1:length(clustersRaw)
        if size(clustersRaw{i},1) >= minClusterSize
            %size(clustersRaw{i},1)
            while size(clustersRaw{i},1) > 4
                %TODO split cluster
                dets = clustersRaw{i};
                dists = squareform(pdist(dets));
                addedDists = sum(dists, 2);
                [~, worstIdx] = max(addedDists);
                idx = 1:size(dets,1);
                clustersRaw{i} = dets(idx ~= worstIdx, :);
            end
            potentialBirds{nClusters+1} = clustersRaw{i};
            nClusters = nClusters + 1;
        elseif size(clustersRaw{i},1) >= 2
            potentialGhosts{nPotentialGhosts+1} = clustersRaw{i};
            nPotentialGhosts = nPotentialGhosts + 1;
        end
    end
    
    if nClusters >= 1
        
        costMatrix = zeros(sum(unassignedPatterns), length(potentialBirds));
        rotMatsMatrix = zeros(sum(unassignedPatterns), length(potentialBirds), 3,3);
        translationsMatrix = zeros(sum(unassignedPatterns), length(potentialBirds), 3);
        unassignedPatternsIdx = find(unassignedPatterns);
        for i = 1:sum(unassignedPatterns)
            for j = 1:length(potentialBirds)
                pattern = squeeze(patterns(unassignedPatternsIdx(i),:,:));
                p = match_patterns(pattern, potentialBirds{j}, 'noKnowledge', params.motionType);
                assignment = zeros(4,1);
                assignment(p) = 1:length(p);
                assignment = assignment(1:size(potentialBirds{j},1));
                pattern = pattern(assignment,:);
                pattern = pattern(assignment > 0, :);
                dets = potentialBirds{j};
                [R, translation, MSE] = umeyama(pattern', dets');
                if size(dets, 1) == 3
                    MSE = MSE*3;
                end
                costMatrix(i,j) = MSE;
                rotMatsMatrix(i,j,:,:) = R;
                translationsMatrix(i,j,:) = translation;
            end
        end
        
        [patternToClusterAssignment, stillUnassignedPatterns, unassignedClusters] = assignDetectionsToTracks(costMatrix, costOfNonAssignment);
        
        %for each (i,j) in patternToClusterAssignment createNewTrack
        for i=1:size(patternToClusterAssignment,1)
            specificPatternIdx = patternToClusterAssignment(i,1);
            clusterIdx = patternToClusterAssignment(i,2);
            pos = squeeze( translationsMatrix(specificPatternIdx, clusterIdx,:) );
            if strcmp(params.model, 'LieGroup')
                rotm = squeeze(rotMatsMatrix(specificPatternIdx, clusterIdx, :,:));
                l2Error = costMatrix(specificPatternIdx, clusterIdx);
                
                % Create a Kalman filter object.
                %TODO adaptive initial Noise!!!!
                patternIdx = unassignedPatternsIdx(specificPatternIdx);
                pattern = squeeze( patterns(patternIdx,:,:));
                newTrack = createLGEKFtrack(rotm, pos, l2Error, patternIdx, pattern, patternNames{patternIdx}, params);
                %             [s, kalmanParams] = setupKalman(pattern, -1, params);
                %             mu.X = [rotm pos; zeros(1,3) 1];
                %             mu.v = zeros(3,1);
                %             mu.a = zeros(3,1);
                %             s.mu = mu;
                %             certaintyFactor = min(l2Error^2, 1);
                %             s.P = diag(repelem([certaintyFactor*params.initialNoise.initQuatVar;
                %                                 certaintyFactor*params.initialNoise.initPositionVar;
                %                                 params.initialNoise.initMotionVar;
                %                                 params.initialNoise.initAccVar
                %                                ],[dim, dim, dim, dim]));
                %             s.pattern = pattern;
                %             s.flying = -1;
                %             s.consecutiveInvisibleCount = 0;
                %             newTrack = struct(...
                %                 'id', patternIdx, ...
                %                 'name', patternNames{patternIdx}, ...
                %                 'kalmanFilter', s, ...
                %                 'kalmanParams', kalmanParams, ...
                %                 'age', 1, ...
                %                 'totalVisibleCount', 1, ...
                %                 'consecutiveInvisibleCount', 0);
                
                % Add it to the array of tracks.
                tracks(patternIdx) = newTrack;
                
                unassignedPatterns(patternIdx) = 0;
                
            else
                quat = rotm2quat( squeeze(rotMatsMatrix(specificPatternIdx, clusterIdx, :,:)) );
                
                % Create a Kalman filter object.
                %TODO adaptive initial Noise!!!!
                patternIdx = unassignedPatternsIdx(specificPatternIdx);
                pattern = squeeze( patterns(patternIdx,:,:));
                [s, kalmanParams] = setupKalman(pattern, -1, params);
                if strcmp(params.quatMotionType, 'brownian')
                    if strcmp(params.motionType, 'constAcc')
                        s.x = [pos; zeros(3,1); zeros(3,1); quat'];
                        % TODO also estimate uncertainty
                        s.P = eye(3*dim+4) .* repelem([kalmanParams.initPositionVar; kalmanParams.initMotionVar; kalmanParams.initAccVar; kalmanParams.initQuatVar], [dim, dim, dim, 4]);
                    else
                        s.x = [pos; zeros(3,1); quat'];
                        % TODO also estimate uncertainty
                        s.P = eye(2*dim+4) .* repelem([kalmanParams.initPositionVar; kalmanParams.initMotionVar; kalmanParams.initQuatVar], [dim, dim, 4]);
                    end
                else
                    if strcmp(params.motionType, 'constAcc')
                        fprintf('not implemented yet')
                    else
                        s.x = [pos'; zeros(3,1); quat'; zeros(4,1)];
                        % TODO also estimate uncertainty
                        s.P = eye(2*dim+8) .* repelem([kalmanParams.initPositionVar; kalmanParams.initMotionVar; kalmanParams.initQuatVar; kalmanParams.initQuatMotionVar], [dim, dim, 4, 4]);
                    end
                end
                s.pattern = pattern;
                newTrack = struct(...
                    'id', patternIdx, ...
                    'name', patternNames{patternIdx}, ...
                    'kalmanFilter', s, ...
                    'kalmanParams', kalmanParams, ...
                    'age', 1, ...
                    'totalVisibleCount', 1, ...
                    'consecutiveInvisibleCount', 0);
                
                % Add it to the array of tracks.
                tracks(patternIdx) = newTrack;
                
                % Increment the next id.
                %nextId = nextId + 1;
                unassignedPatterns(patternIdx) = 0;
            end
            
        end
        
        positions = 10000*ones(length(tracks)+length(ghostTracks),3);
        for i=1:length(tracks)
            if tracks(i).age > 0
                positions(i,:) = tracks(i).kalmanFilter.mu.X(1:3, 4);
            end
        end
        for i=1:length(ghostTracks)
           positions(length(tracks)+i, :) = ghostTracks(i).kalmanFilter.x(1:3); 
        end
        
        % for each unassigned cluster create a ghost bird
        for i=1:size(unassignedClusters)
            pos = mean(potentialBirds{unassignedClusters(i)}, 1);
            dists = pdist2(pos, positions);
            if min(dists, [], 2) > minDistToBird
                kF = constructGhostKF(pos, params);
                ghostTrack = struct(...
                    'kalmanFilter', kF, ...
                    'age', 1, ...
                    'totalVisibleCount', 1, ...
                    'consecutiveInvisibleCount', 0);
                ghostTracks(length(ghostTracks) + 1) = ghostTrack;
            end
        end
    end
        
    %for each potential Ghost create a ghost bird
    if exist('positions','var') ~= 1
        positions = 10000*ones(length(tracks)+length(ghostTracks),3);
        for i=1:length(tracks)
            if tracks(i).age > 0
                positions(i,:) = tracks(i).kalmanFilter.mu.X(1:3, 4);
            end
        end
        for i=1:length(ghostTracks)
           positions(length(tracks)+i, :) = ghostTracks(i).kalmanFilter.x(1:3); 
        end
    end
    for i=1:size(potentialGhosts)
        pos = mean(potentialGhosts{i}, 1);
        dists = pdist2(pos, positions);
        if min(dists, [], 2) > minDistToBird
            kF = constructGhostKF(pos, params);
            ghostTrack = struct(...
                'kalmanFilter', kF, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);
            ghostTracks(length(ghostTracks) + 1) = ghostTrack;
        end
    end
end

end

function kF = constructGhostKF(pos, params)
% setup simple Kalman filter, with position, velocity and
% acceleration
% the state transition function implements a
% constant-acceleration-motion-model
% the measurement function is selects the position from the state
% and measurements are the mean of all assigne detections

kF.x = pos';
kF.P = eye(3) .* repelem(params.initialNoise.initPositionVar, 3);
kF.Q = eye(3) .* repelem(params.processNoise.position, 3);
kF.R = eye(3) * params.measurementNoise/2;
kF.H = eye(3);
kF.F = eye(3);


% kF.x = [pos 0 0 0 0 0 0]';
% kF.P = eye(9) .* repelem([params.initialNoise.initPositionVar;
%     params.initialNoise.initMotionVar;
%     params.initialNoise.initAccVar], 3);
% kF.Q = eye(9) .* repelem([params.processNoise.position;
%     params.processNoise.motion;
%     params.processNoise.acceleration], 3);
% kF.R = eye(3) * params.measurementNoise*5;
% kF.H = [eye(3) zeros(3,6)];
% kF.F = [eye(3) eye(3) 1/2*eye(3);
%     zeros(3) eye(3) eye(3);
%     zeros(3) zeros(3) eye(3)];
end

