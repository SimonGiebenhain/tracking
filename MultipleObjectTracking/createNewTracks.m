function [tracks, ghostTracks, unassignedPatternsReturn] = createNewTracks(detections, unassignedPatternsReturn, tracks, patterns, params, patternNames, similarPairs, ghostTracks)
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

% Ghost birds should be initialized in the proximity of other birds
minDistToBird = params.minDistToBird;%90;
initThreshold = params.initThreshold;%1.25;



% birds which were invisible for the last 5 frames or more are allowed to
% be re-initialized
invisCounts = [tracks(:).consecutiveInvisibleCount];
invis = invisCounts >= 5;
unassignedPatterns = unassignedPatternsReturn | invis';

if size(detections, 1) > 1 && sum(unassignedPatterns) > 0
    costOfNonAssignment = 1;
    
    dim = size(patterns,3);
    epsilon = 55;
    clustersRaw = clusterUnassignedDetections(detections, epsilon);
    nClusters4 = 0;
    nClusters3 = 0;
    nPotentialGhosts = 0;
    potentialBirds3 = {};
    potentialBirds4 = {};
    potentialGhosts = {};
    for i=1:length(clustersRaw)
        if size(clustersRaw{i},1) >= 4
            %size(clustersRaw{i},1)
            while size(clustersRaw{i},1) > 4
                dets = clustersRaw{i};
                dists = squareform(pdist(dets));
                addedDists = sum(dists, 2);
                [~, worstIdx] = max(addedDists);
                idx = 1:size(dets,1);
                clustersRaw{i} = dets(idx ~= worstIdx, :);
            end
            potentialBirds4{nClusters4+1} = clustersRaw{i};
            nClusters4 = nClusters4 + 1;
        elseif size(clustersRaw{i},1) == 3
            potentialBirds3{nClusters3+1} = clustersRaw{i};
            nClusters3 = nClusters3 + 1;
        elseif size(clustersRaw{i},1) == 2
            potentialGhosts{nPotentialGhosts+1} = clustersRaw{i};
            nPotentialGhosts = nPotentialGhosts + 1;
        end
    end
    
    if nClusters4 + nClusters3 >= 1
        costMatrix = zeros(sum(unassignedPatterns), length(potentialBirds4));
        rotMatsMatrix = zeros(sum(unassignedPatterns), length(potentialBirds4), 3,3);
        translationsMatrix = zeros(sum(unassignedPatterns), length(potentialBirds4), 3);
        unassignedPatternsIdx = find(unassignedPatterns);
        for i = 1:sum(unassignedPatterns)
            for j = 1:length(potentialBirds4)
                pattern = squeeze(patterns(unassignedPatternsIdx(i),:,:));
                p = match_patterns(pattern, potentialBirds4{j}, 'noKnowledge', params.motionType);
                assignment = zeros(4,1);
                assignment(p) = 1:length(p);
                assignment = assignment(1:size(potentialBirds4{j},1));
                pattern = pattern(assignment,:);
                pattern = pattern(assignment > 0, :);
                dets = potentialBirds4{j};
                [R, translation, MSE] = umeyama(pattern', dets');
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
                patternIdx = unassignedPatternsIdx(specificPatternIdx);
                pattern = squeeze( patterns(patternIdx,:,:));
                newTrack = createLGEKFtrack(rotm, pos, l2Error, patternIdx, pattern, patternNames{patternIdx}, params);
                
                % Add it to the array of tracks.
                tracks(patternIdx) = newTrack;
                
                unassignedPatterns(patternIdx) = 0;
                unassignedPatternsReturn(patternIdx) = 0;
                
            else
                quat = rotm2quat( squeeze(rotMatsMatrix(specificPatternIdx, clusterIdx, :,:)) );
                
                % Create a Kalman filter object.
                patternIdx = unassignedPatternsIdx(specificPatternIdx);
                pattern = squeeze( patterns(patternIdx,:,:));
                [s, kalmanParams] = setupKalman(pattern, -1, params);
                if strcmp(params.quatMotionType, 'brownian')
                    if strcmp(params.motionType, 'constAcc')
                        s.x = [pos; zeros(3,1); zeros(3,1); quat'];
                        s.P = eye(3*dim+4) .* repelem([kalmanParams.initPositionVar; kalmanParams.initMotionVar; kalmanParams.initAccVar; kalmanParams.initQuatVar], [dim, dim, dim, 4]);
                    else
                        s.x = [pos; zeros(3,1); quat'];
                        s.P = eye(2*dim+4) .* repelem([kalmanParams.initPositionVar; kalmanParams.initMotionVar; kalmanParams.initQuatVar], [dim, dim, 4]);
                    end
                else
                    if strcmp(params.motionType, 'constAcc')
                        fprintf('not implemented yet')
                    else
                        s.x = [pos'; zeros(3,1); quat'; zeros(4,1)];
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
                unassignedPatternsReturn(patternIdx) = 0;
            end
            
        end
        
        %TODO pack all stuff regarding patterns in one struct
        % this way safePatterns does not have to be re-calculated as
        % frequently

        if nClusters3 > 0
            assignedPatternsIdx = find(~unassignedPatterns);
            unassignedPatternsIdx = find(unassignedPatterns);
            safePatternsBool = zeros(length(patterns), 1);
            % Determine patterns without unassigned, similar patterns
            for p=1:length(unassignedPatternsIdx)
                patIdx = unassignedPatternsIdx(p);
                conflicts = similarPairs(similarPairs(:, 1) == patIdx, 2);
                conflicts = [conflicts; similarPairs(similarPairs(:, 2) == patIdx, 1)];
                conflicts = setdiff(conflicts, assignedPatternsIdx);
                if isempty(conflicts) == 0
                    safePatternsBool(patIdx) = 1;
                end
            end
            
            safeAndUnassignedPatterns = find(unassignedPatterns & safePatternsBool);
            nSafeAndUnassigned = length(safeAndUnassignedPatterns);
            
            deletedClusters3 = zeros(nClusters3,1);
            for c=1:nClusters3
                costs = 1000*ones(nSafeAndUnassigned, 1);
                rotMats = zeros(nSafeAndUnassigned, 3, 3);
                transVecs = zeros(nSafeAndUnassigned, 3);
                
                dets = potentialBirds3{c};
                for jj=1:nSafeAndUnassigned
                    p = safeAndUnassignedPatterns(jj);
                    pattern = squeeze(patterns(p,:,:));
                    permut = match_patterns(pattern, dets, 'noKnowledge', params.motionType);
                    assignment = zeros(4,1);
                    assignment(permut) = 1:length(permut);
                    assignment = assignment(1:size(dets,1));
                    pattern = pattern(assignment,:);
                    pattern = pattern(assignment > 0, :);
                    [R, translation, MSE] = umeyama(pattern', dets');
                    costs(jj) = MSE;
                    rotMats(jj,:,:) = R;
                    transVecs(jj,:) = translation;
                end
                
                [minMSE, minIdx] = min(costs);
                minPatIdx = safeAndUnassignedPatterns(minIdx);
                
                if minMSE < initThreshold
                    newTrack = createLGEKFtrack(squeeze(rotMats(minIdx, :, :)), ...
                        squeeze(transVecs(minIdx, :))', ...
                        minMSE, minPatIdx, ...
                        squeeze(patterns(minPatIdx, :, :)),...
                        patternNames{minPatIdx}, params);
                    tracks(minPatIdx) = newTrack;
                    unassignedPatterns(minPatIdx) = 0;
                    unassignedPatternsReturn(minPatIdx) = 0;
                    deletedClusters3(c) = 1;
                end
            end
            nClusters3 = nClusters3 - nnz(deletedClusters3);
            potentialBirds3(deletedClusters3 == 1) = [];
            
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
        
        % for each unassigned cluster of size 4 create a ghost bird
        for i=1:size(unassignedClusters)
            pos = mean(potentialBirds4{unassignedClusters(i)}, 1);
            dists = pdist2(pos, positions);
            if min(dists, [], 2) > minDistToBird
                kF = constructGhostKF(pos, params);
                ghostTrack = struct(...
                    'kalmanFilter', kF, ...
                    'age', 1, ...
                    'totalVisibleCount', 1, ...
                    'consecutiveInvisibleCount', 0);
                ghostTracks(length(ghostTracks) + 1) = ghostTrack;
                positions(length(positions)+1, :) = pos;
            end
        end
        
        % for each unassigned cluster of size 3 create a ghost bird
        for i=1:nClusters3
            pos = mean(potentialBirds3{i}, 1);
            dists = pdist2(pos, positions);
            if min(dists, [], 2) > minDistToBird
               kF = constructGhostKF(pos, params);
                ghostTrack = struct(...
                    'kalmanFilter', kF, ...
                    'age', 1, ...
                    'totalVisibleCount', 1, ...
                    'consecutiveInvisibleCount', 0);
                ghostTracks(length(ghostTracks) + 1) = ghostTrack;
                positions(length(positions)+1, :) = pos;
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

