function [tracks, unassignedPatterns] = createNewTracks(detections, unassignedPatterns, tracks, patterns, params)
%CREATENEWTRACKS Summary of this function goes here
%   Detailed explanation goes here

if length(detections) > 1 && sum(unassignedPatterns) > 0    
    dim = size(patterns,3);
    epsilon = 50;
    clustersRaw = clusterUnassignedDetections(detections, epsilon);
    nClusters = 0;
    clusters = {};
    for i=1:length(clustersRaw)
        %TODO what to do with less than 4 detections
        if size(clustersRaw{i},1) == 4
            if size(clustersRaw{i},1) > 4
                %TODO remove markers
            end
            clusters{nClusters+1} = clustersRaw{i};
            nClusters = nClusters + 1;
        end
    end
    
    if nClusters < 1
        return
    end
    costMatrix = zeros(sum(unassignedPatterns), length(clusters));
    rotMatsMatrix = zeros(sum(unassignedPatterns), length(clusters), 3,3);
    translationsMatrix = zeros(sum(unassignedPatterns), length(clusters), 3);
    unassignedPatternsIdx = find(unassignedPatterns);
    for i = 1:sum(unassignedPatterns)
        for j = 1:length(clusters)
            pattern = squeeze(patterns(unassignedPatternsIdx(i),:,:));
            p = match_patterns(pattern, clusters{j}, 'noKnowledge');
            assignment = zeros(4,1);
            assignment(p) = 1:length(p);
            pattern = pattern(assignment,:);
            dets = clusters{j};
            %size(dets,1)
            % TODO augment missing detection
            [R, translation, MSE] = umeyama(pattern', dets');
            costMatrix(i,j) = MSE;
            rotMatsMatrix(i,j,:,:) = R;
            translationsMatrix(i,j,:) = translation;
        end
    end
    
    costOfNonAssignment = 2; %TODO find something reasonable, altough could be very high
    [patternToClusterAssignment, stillUnassignedPatterns, ~] = ...
        assignDetectionsToTracks(costMatrix, costOfNonAssignment);
    
    %for each (i,j) in patternToClusterAssignment createNewTrack
    for i=1:size(patternToClusterAssignment,1)
        specificPatternIdx = patternToClusterAssignment(i,1);
        clusterIdx = patternToClusterAssignment(i,2);
        pos = squeeze( translationsMatrix(specificPatternIdx, clusterIdx,:) );
        quat = rotm2quat( squeeze(rotMatsMatrix(specificPatternIdx, clusterIdx, :,:)) );
        
        % Create a Kalman filter object.
        %TODO adaptive initial Noise!!!!
        patternIdx = unassignedPatternsIdx(specificPatternIdx);
        pattern = squeeze( patterns(patternIdx,:,:));
        [s, kalmanParams] = setupKalman(pattern, -1, params.model, params.quatMotionType, params.measurementNoise, params.processNoise, params.initialNoise);
        if strcmp(params.quatMotionType, 'brownian')
            s.x = [pos; zeros(3,1); quat'];
            % TODO also estimate uncertainty
            s.P = eye(2*dim+4) .* repelem([kalmanParams.initPositionVar; kalmanParams.initMotionVar; kalmanParams.initQuatVar], [dim, dim, 4]);
        else
            s.x = [pos'; zeros(3,1); quat'; zeros(4,1)];
            % TODO also estimate uncertainty
            s.P = eye(2*dim+8) .* repelem([kalmanParams.initPositionVar; kalmanParams.initMotionVar; kalmanParams.initQuatVar; kalmanParams.initQuatMotionVar], [dim, dim, 4, 4]);
        end
        s.pattern = pattern;
        newTrack = struct(...
            'id', patternIdx, ...
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

end

