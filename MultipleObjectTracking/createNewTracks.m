function [tracks, ghostTracks, unassignedPatternsReturn, nextGhostTrackID, freshInits] = createNewTracks(detections, unassignedPatternsReturn, tracks, patterns, params, patternNames, similarPairs, lastVisibleFrames, ghostTracks, nextGhostTrackID)
%CREATENEWTRACKS Search for patterns of untracked birds in unassigned
%detections in order to initialize new tracks.
%   Arguments:
%   @detections array of dimensions [N x 3], where N is the number of
%   detections which have not been assigned to one of the tracked birds
%   @unassignedPatternsReturn binary array, where a 1 in the i-th position
%   denotes, that the i-th bird is not currently tracked
%   @tracks struct array, where each element holds all relevant information
%   about a tracked birds
%   @patterns array of dimensions [K x 4 x 3], holding the patterns for
%   each of the K birds of the recording. Each pattern is specified by 4 
%   3-D points
%   @params struct holding several parameters used inside the Kalman filter 
%   as well as in the initialization procedure
%   @patternNames struct holding the names of the patterns
%   @similarPairs array of dimensions [M x 2], where each row contains the
%   pattern indices of two patterns which are similar to each other.
%   @lastVisibleFrames lists frames when each bird was last seen
%   @ghostTracks struct array, where each struct holds all relevant 
%   information about a ghost track
%   @nextGhostTrackID number indicating the ID of the next ghostTrack to be
%   initialized. 
%
%   Returns:
%   @tracks updated version of trakcs
%   @ghostTracks updated version of ghostTracks
%   @unassignedPatternsReturn list of IDs of birds that are still untracked
%   @nextGhostTrackID ghostTrackID to be used next
%   @freshInits binary array indicating which birds are still not tracked

freshInits = zeros(length(tracks),1);

if exist('ghostTracks','var') ~= 1
    kF = constructGhostKF([0 0 0], params);
    ghostTrack = struct(...
        'ID', 0, ...
        'kalmanFilter', kF, ...
        'age', 1, ...
        'totalVisibleCount', 1, ...
        'consecutiveInvisibleCount', 0, ...
        'trustworthyness', 0, ...
        'potentialInits', zeros(size(patterns,1), 1), ...
        'lastPotentialInit', -100);
    ghostTracks(1) = ghostTrack;
    ghostTracks(:, 1) = [];
    nextGhostTrackID = 1;
end

% Ghost birds should be initialized in the proximity of other birds
minDistToBird = params.minDistToBird;

% Get positions of all active tracks
ages = [tracks(:).age];
nActiveTracks = sum(ages>0);
trackedPos = zeros(nActiveTracks, 3);
idx = 1;
for i=1:length(tracks)
    if tracks(i).age > 0
       trackedPos(idx, :) =  tracks(i).kalmanFilter.mu.X(1:3, 4);
       idx = idx + 1;
    end
end



% birds which were invisible for the last 5 frames or more are allowed to
% be re-initialized
invisCounts = [tracks(:).consecutiveInvisibleCount];
invis = invisCounts >= 5;
unassignedPatterns = unassignedPatternsReturn | invis';

if size(detections, 1) > 1 && sum(unassignedPatterns) > 0    
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
       deletedClusters4 = zeros(length(potentialBirds4));
        for j=1:length(potentialBirds4)
            costs = zeros(sum(unassignedPatterns), 1);
            rots = zeros(sum(unassignedPatterns), 3, 3);
            poss = zeros(sum(unassignedPatterns), 3);
            unassignedPatternsIdx = find(unassignedPatterns);
            for i=1:sum(unassignedPatterns)
                pattern = squeeze(patterns(unassignedPatternsIdx(i),:,:));
                p = match_patterns(pattern, potentialBirds4{j}, 'noKnowledge', params.motionType);
                assignment = zeros(4,1);
                assignment(p) = 1:length(p);
                assignment = assignment(1:size(potentialBirds4{j},1));
                pattern = pattern(assignment,:);
                pattern = pattern(assignment > 0, :);
                dets = potentialBirds4{j};
                [R, translation, MSE] = umeyama(pattern', dets');
                costs(i) = MSE;
                rots(i, :, :) = R;
                poss(i, :, :) = translation;
            end
            
            % select min and initialize
            [minCosts2, minIdx2] = mink(costs, 2);
            if length(minCosts2) == 1
                costDiff = Inf;
            elseif length(minCosts2) == 2
                costDiff = minCosts2(2) - minCosts2(1);
            else
                'ERROR';
            end
            if minCosts2(1) < params.initThreshold4Tight && costDiff > params.costDiff4Tight
                % Check whether new bird is too close to existing
                        % bird!
                specificPatternIdx = minIdx2(1);
                pos = squeeze( poss(specificPatternIdx,:) );
                dists = pdist2(pos, trackedPos);
                if min(dists) < 75
                    continue;
                end
                        
                
                deletedClusters4(j) = 1;
                if strcmp(params.model, 'LieGroup')
                    rotm = squeeze(rots(specificPatternIdx,:,:));
                    l2Error = minCosts2(1);
                    
                    % Create a Kalman filter object.
                    patternIdx = unassignedPatternsIdx(specificPatternIdx);
                    pattern = squeeze( patterns(patternIdx,:,:));
                    if isfield(tracks(patternIdx).kalmanFilter, 'mu')
                        newTrack = createLGEKFtrack(rotm, pos', l2Error, patternIdx, pattern, patternNames{patternIdx}, params, tracks(patternIdx).kalmanFilter.mu.motionModel);
                    else
                        newTrack = createLGEKFtrack(rotm, pos', l2Error, patternIdx, pattern, patternNames{patternIdx}, params);
                    end
                    
                    newTrack.kalmanFilter.lastVisibleFrame = lastVisibleFrames(patternIdx);
                    % Add it to the array of tracks.
                    tracks(patternIdx) = newTrack;
                    
                    unassignedPatterns(patternIdx) = 0;
                    unassignedPatternsReturn(patternIdx) = 0;
                    freshInits(patternIdx) = true;
                    
                else
                    quat = rotm2quat( squeeze(rots(specificPatternIdx, :,:)) );
                    
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
                    
                    newTrack.kalmanFilter.lastVisibleFrame = lastVisibleFrames(patternIdx);
                    % Add it to the array of tracks.
                    tracks(patternIdx) = newTrack;
                    
                    % Increment the next id.
                    %nextId = nextId + 1;
                    unassignedPatterns(patternIdx) = 0;
                    unassignedPatternsReturn(patternIdx) = 0;
                    
                    freshInits(patternIdx) = true;
                end
            end
        end
        
        nClusters4 = nClusters4 - nnz(deletedClusters4);
        potentialBirds4(deletedClusters4 == 1) = [];
        
        %[patternToClusterAssignment, stillUnassignedPatterns, unassignedClusters] = assignDetectionsToTracks(costMatrix, costOfNonAssignment);

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
                if isempty(conflicts)
                    safePatternsBool(patIdx) = 1;
                end
            end
            
            safeAndUnassignedPatterns = find(unassignedPatterns & safePatternsBool);
            nSafeAndUnassigned = length(safeAndUnassignedPatterns);
            if nSafeAndUnassigned > 0
            
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

                    [minMSE2, minIdx2] = mink(costs, 2);
                    if length(minMSE2) == 2
                        costDiff = minMSE2(2) - minMSE2(1);
                    elseif length(minMSE2) == 1
                        costDiff = Inf;
                    end
                    minPatIdx = safeAndUnassignedPatterns(minIdx2(1));

                    if minMSE2(1) < params.initThresholdTight && costDiff > params.costDiffTight
                        % Check whether new bird is too close to existing
                        % bird!
                        pos = squeeze(transVecs(minIdx2(1), :))';
                        dists = pdist2(pos', trackedPos);
                        if min(dists) < 75
                            continue;
                        end
                        
                        if isfield(tracks(minPatIdx).kalmanFilter, 'mu')
                            newTrack = createLGEKFtrack(squeeze(rotMats(minIdx2(1), :, :)), ...
                                pos, ...
                                minMSE2(1), minPatIdx, ...
                                squeeze(patterns(minPatIdx, :, :)),...
                                patternNames{minPatIdx}, params, ...
                                tracks(minPatIdx).kalmanFilter.mu.motionModel);
                        else
                            newTrack = createLGEKFtrack(squeeze(rotMats(minIdx2(1), :, :)), ...
                                squeeze(transVecs(minIdx2(1), :))', ...
                                minMSE2(1), minPatIdx, ...
                                squeeze(patterns(minPatIdx, :, :)),...
                                patternNames{minPatIdx}, params);
                        end
                        newTrack.kalmanFilter.lastVisibleFrame = lastVisibleFrames(minPatIdx);
                        tracks(minPatIdx) = newTrack;
                        unassignedPatterns(minPatIdx) = 0;
                        unassignedPatternsReturn(minPatIdx) = 0;
                        freshInits(minPatIdx) = true;
                        deletedClusters3(c) = 1;
                    end
                end
                nClusters3 = nClusters3 - nnz(deletedClusters3);
                potentialBirds3(deletedClusters3 == 1) = [];
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
        
        % for each unassigned cluster of size 4 create a ghost bird
        for i=1:nClusters4
            dists = pdist2(potentialBirds4{i}, positions);
            if min(dists, [], 'all') > minDistToBird(4)
                pos = mean(potentialBirds4{i}, 1);
                kF = constructGhostKF(pos, params);
                ghostTrack = struct(...
                    'ID', nextGhostTrackID, ...
                    'kalmanFilter', kF, ...
                    'age', 1, ...
                    'totalVisibleCount', 1, ...
                    'consecutiveInvisibleCount', 0, ...
                    'trustworthyness', 12, ...
                    'potentialInits', zeros(size(patterns,1), 1), ...
                    'lastPotentialInit', -100);
                ghostTracks(length(ghostTracks) + 1) = ghostTrack;
                positions(length(positions)+1, :) = pos;
                nextGhostTrackID = nextGhostTrackID + 1;
            end
        end
        
        % for each unassigned cluster of size 3 create a ghost bird
        for i=1:nClusters3
            dists = pdist2(potentialBirds3{i}, positions);
            if min(dists, [], 'all') > minDistToBird(3)
               pos = mean(potentialBirds3{i}, 1);
               kF = constructGhostKF(pos, params);
                ghostTrack = struct(...
                    'ID', nextGhostTrackID, ...
                    'kalmanFilter', kF, ...
                    'age', 1, ...
                    'totalVisibleCount', 1, ...
                    'consecutiveInvisibleCount', 0, ...
                    'trustworthyness', 5, ...
                    'potentialInits', zeros(size(patterns,1), 1), ...
                    'lastPotentialInit', -100);
                ghostTracks(length(ghostTracks) + 1) = ghostTrack;
                positions(length(positions)+1, :) = pos;
                nextGhostTrackID = nextGhostTrackID + 1;

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
        dists = pdist2(potentialGhosts{i}, positions);
        if min(dists, [], 'all') > minDistToBird(2)
            pos = mean(potentialGhosts{i}, 1);
            kF = constructGhostKF(pos, params);
            ghostTrack = struct(...
                'ID', nextGhostTrackID, ...
                'kalmanFilter', kF, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0, ...
                'trustworthyness', 1, ...
                'potentialInits', zeros(size(patterns,1), 1), ...
                'lastPotentialInit', -100);
            ghostTracks(length(ghostTracks) + 1) = ghostTrack;
            nextGhostTrackID = nextGhostTrackID + 1;

        end
    end
end

end

function kF = constructGhostKF(pos, params, ghostMM)
% setup simple Kalman filter, with position, velocity and
% acceleration
% the state transition function implements a
% constant-acceleration-motion-model
% the measurement function is selects the position from the state
% and measurements are the mean of all assigne detections
if ~exist('ghostMM', 'var')
   ghostMM = 0; 
end
if ghostMM == 0
    kF.motionModel = 0;
    kF.x = pos';
    kF.P = eye(3) .* repelem(params.initialNoise.initPositionVar, 3);
    kF.Q = eye(3) .* repelem(params.processNoise.position, 3);
    kF.R = eye(3) * params.measurementNoise/2;
    kF.H = eye(3);
    kF.F = eye(3);
elseif ghostMM == 1
    kF.motionModel = 1;
    kF.x = [pos 0 0 0]';
    kF.P = eye(6) .* repelem([10; 5], 3);
    kF.Q = eye(6) .* repelem([50, 10], 3);
    kF.R = eye(3) * 120;
    kF.H = [eye(3) zeros(3)];
    kF.F = [eye(3) eye(3);
            zeros(3) eye(3)];
elseif ghostMM == 2
    kF.motionModel = 2;
    kF.x = [pos 0 0 0 0 0 0]';
    kF.P = eye(9) .* repelem([10; 5; 2], 3);
    kF.Q = eye(9) .* repelem([50; 5; 1], 3);
    kF.R = eye(3) * 120;
    kF.H = [eye(3) zeros(3,6)];
    kF.F = [eye(3) eye(3) 1/2*eye(3);
            zeros(3) eye(3) eye(3);
            zeros(3) zeros(3) eye(3)];
end
kF.framesInMotionModel = 15;
kF.latest5pos = zeros(5,3);
kF.latest5pos(1, :) = pos;
kF.latestPosIdx = 1;
end

