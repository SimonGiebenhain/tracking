function [tracks, unassignedPatterns] = createNewTracks(detections, unassignedPatterns, tracks, patterns, params)
%CREATENEWTRACKS Summary of this function goes here
%   Detailed explanation goes here

if size(detections, 1) > 1 && sum(unassignedPatterns) > 0  
    %fprintf('Creatng new tracks')
    dim = size(patterns,3);
    epsilon = 55;
    clustersRaw = clusterUnassignedDetections(detections, epsilon);
    nClusters = 0;
    clusters = {};
    for i=1:length(clustersRaw)
        if size(clustersRaw{i},1) > 3
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
            p = match_patterns(pattern, clusters{j}, 'noKnowledge', params.motionType);
            assignment = zeros(4,1);
            assignment(p) = 1:length(p);
            assignment = assignment(1:size(clusters{j},1));
            pattern = pattern(assignment,:);
            pattern = pattern(assignment > 0, :);
            dets = clusters{j};
            [R, translation, MSE] = umeyama(pattern', dets');
            costMatrix(i,j) = MSE;
            rotMatsMatrix(i,j,:,:) = R;
            translationsMatrix(i,j,:) = translation;
        end
    end
    
    costOfNonAssignment = 2; 
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

