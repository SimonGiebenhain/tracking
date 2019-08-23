%% Multi Object Tracking
% The original code structure comes from the MATLAB tutorial for motion
% based multi object tracking: 
% https://de.mathworks.com/help/vision/examples/motion-based-multiple-object-tracking.html

% TODO
% Explanation goes here

function [estimatedPositions, estimatedQuats] = ownMOT(D, patterns, initialStates, nObjects, trueTrajectory, trueOrientation, quatMotionType)
% OWNMOT does multi object tracking
%   @D all observations/detections in the format:
%       T x maxDetectionsPerFrame x 3
%
%   @patterns array of dimensions nObjects x nMarkers x 3
%       Each object has a unique pattern of nMarker 3d points
%
%   @initialStates values used for the initialization of EKF.
%       TODO: use own initialzation method instead of values from VICON.
%
%   @nObject the number of objects to track
%
%   @trueTrajectory array of dimensions nObjects x T x 3
%       Holds ground truth trajectory. When supplied this is used for the
%       visualization.
%
%   @trueOrientation array of dimensions nObjects x T x 4
%       Holds ground truth quaternions representing the orientation of each
%       object at each timeframe

nMarkers = 4;

[T, ~, dim] = size(D);

maxPos = squeeze(max(D,[],[1 2]));
minPos = squeeze(min(D,[],[1 2]));

processNoise.position = 20;
processNoise.motion = 20;
processNoise.quat = 0.02;
processNoise.quatMotion = 0.02;
measurementNoise = 150;
model = 'extended';
initialNoise.initPositionVar = 5;
initialNoise.initMotionVar = 5;
initialNoise.initQuatVar = 0.2;
initialNoise.initQuatMotionVar = 0.2;

nextId = 1;
tracks = initializeTracks();
unassignedPatterns = zeros(nObjects, 1);

markersForVisualization = cell(nObjects,1);
birdsTrajectories = cell(nObjects,1);
trueTrajectories = cell(nObjects,1);
birdsPositions = cell(nObjects,1);
markerPositions = cell(nObjects, nMarkers);
viconMarkerPositions = cell(nObjects, nMarkers);

colorsPredicted = distinguishable_colors(nObjects);
colorsTrue = (colorsPredicted + 2) ./ (max(colorsPredicted,[],2) +2);
keepOldTrajectory = 0;
shouldShowTruth = 1;
vizHistoryLength = 200;
initializeFigure();


estimatedPositions = zeros(nObjects, T, 3);
estimatedQuats = zeros(nObjects, T, 4);


for t = 1:T
    detections = squeeze(D(t,:,:));
    detections = reshape(detections(~isnan(detections)),[],dim);
    
    predictNewLocationsOfTracks();
    [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignment();
    
    updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks();

    createNewTracks();
    if t == 1020
       t 
    end
    t
    %if t > 9000
        displayTrackingResults();
    %end
    
    % Store tracking results
    for ii = 1:nObjects
        if tracks(ii).age > 0
            state = tracks(ii).kalmanFilter.x;
            estimatedPositions(ii,t,:) = state(1:dim);
            estimatedQuats(ii,t,:) = state(2*dim+1:2*dim+4);
        else
            estimatedPositions(ii,t,:) = ones(3,1) * NaN;
            estimatedQuats(ii,t,:) = ones(4,1) * NaN;
        end
    end
    
end



%% Initialize Tracks
%{
% The |initializeTracks| function creates an array of tracks, where each
% track is a structure representing a moving object in the video. The
% purpose of the structure is to maintain the state of a tracked object.
% The state consists of information used for detection to track assignment,
% track termination, and display.
%
% The structure contains the following fields:
%
% * |id| :                  the integer ID of the track
% * |bbox| :                the current bounding box of the object; used
%                           for display
% * |kalmanFilter| :        a Kalman filter object used for motion-based
%                           tracking
% * |age| :                 the number of frames since the track was first
%                           detected
% * |totalVisibleCount| :   the total number of frames in which the track
%                           was detected (visible)
% * |consecutiveInvisibleCount| : the number of consecutive frames for
%                                  which the track was not detected (invisible).
%
% Noisy detections tend to result in short-lived tracks. For this reason,
% the example only displays an object after it was tracked for some number
% of frames. This happens when |totalVisibleCount| exceeds a specified
% threshold.
%
% When no detections are associated with a track for several consecutive
% frames, the example assumes that the object has left the field of view
% and deletes the track. This happens when |consecutiveInvisibleCount|
% exceeds a specified threshold. A track may also get deleted as noise if
% it was tracked for a short time, and marked invisible for most of the
% frames.
%}

    function tracks = initializeTracks()
        
        for i = 1:nObjects
            [s, kalmanParams] = setupKalman(squeeze(patterns(i,:,:)), -1, model, quatMotionType, measurementNoise, processNoise, initialNoise);
            if strcmp(quatMotionType, 'brownian')
                s.x = initialStates(i,1:end-4)';
                s.P = eye(2*dim+4) .* repelem([kalmanParams.initPositionVar; kalmanParams.initMotionVar; kalmanParams.initQuatVar], [dim, dim, 4]);

            else
                s.x = initialStates(i,:)';
                s.P = eye(2*dim+8) .* repelem([kalmanParams.initPositionVar; kalmanParams.initMotionVar; kalmanParams.initQuatVar; kalmanParams.initQuatMotionVar], [dim, dim, 4, 4]);
            end
            s.pattern = squeeze(patterns(i,:,:));
            tracks(i) = struct(...
                'id', nextId, ... %'center', , ...
                'kalmanFilter', s, ...
                'kalmanParams', kalmanParams, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);
            nextId = nextId + 1;
        end
    end

%% Predict New Locations of Existing Tracks
% Use the Kalman filter to predict the centroid of each track in the
% current frame, and update its bounding box accordingly.

    function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            if tracks(i).age > 0
                % Predict the current location of the track.
                tracks(i).kalmanFilter = predictKalman(tracks(i).kalmanFilter, 1, tracks(i).kalmanParams, 'extended');
            end
        end
    end

%% Assign Detections to Tracks
%{
% Assigning object detections in the current frame to existing tracks is
% done by minimizing cost. The cost is defined as the negative
% log-likelihood of a detection corresponding to a track.
%
% The algorithm involves two steps:
%
% Step 1: Compute the cost of assigning every detection to each track using
% the |distance| method of the |vision.KalmanFilter| System object(TM). The
% cost takes into account the Euclidean distance between the predicted
% centroid of the track and the centroid of the detection. It also includes
% the confidence of the prediction, which is maintained by the Kalman
% filter. The results are stored in an MxN matrix, where M is the number of
% tracks, and N is the number of detections.
%
% Step 2: Solve the assignment problem represented by the cost matrix using
% the |assignDetectionsToTracks| function. The function takes the cost
% matrix and the cost of not assigning any detections to a track.
%
% The value for the cost of not assigning a detection to a track depends on
% the range of values returned by the |distance| method of the
% |vision.KalmanFilter|. This value must be tuned experimentally. Setting
% it too low increases the likelihood of creating a new track, and may
% result in track fragmentation. Setting it too high may result in a single
% track corresponding to a series of separate moving objects.
%
% The |assignDetectionsToTracks| function uses the Munkres' version of the
% Hungarian algorithm to compute an assignment which minimizes the total
% cost. It returns an M x 2 matrix containing the corresponding indices of
% assigned tracks and detections in its two columns. It also returns the
% indices of tracks and detections that remained unassigned.
%}

    function [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignment()
        
        nTracks = length(tracks);
        nDetections = size(detections, 1);
        
        % Compute the cost of assigning each detection to each marker.
        cost = zeros(nTracks*nMarkers, nDetections);
        for i = 1:nTracks
            if tracks(i).age > 0
                cost((i-1)*nMarkers+1:i*nMarkers, :) = distanceKalman(tracks(i).kalmanFilter, detections);
            else
                cost((i-1)*nMarkers+1:i*nMarkers, :) = Inf;
            end
        end
        
        % Solve the assignment problem.
        costOfNonAssignment = 50;
        [assignments, unassignedTracks, unassignedDetections] = assignDetectionsToTracks(cost, costOfNonAssignment);
    end


%% Update Assigned Tracks
%{
% The |updateAssignedTracks| function updates each assigned track with the
% corresponding detection. It calls the |correct| method of
% |vision.KalmanFilter| to correct the location estimate. Next, it stores
% the new bounding box, and increases the age of the track and the total
% visible count by 1. Finally, the function sets the invisible count to 0.
%}

    function updateAssignedTracks()
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % TOOOOODDDDDDOOOOOOO
        % handle lost tracks, i.e. age == 0 !!!!
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        
        assignments = double(assignments);
        allAssignedTracksIdx = unique(floor((assignments(:,1)-1)/nMarkers) + 1);
        nAssignedTracks = length(allAssignedTracksIdx);
        for i = 1:nAssignedTracks
            currentTrackIdx = allAssignedTracksIdx(i);
            assignmentsIdx = floor((assignments(:,1)-1)/nMarkers) + 1 == currentTrackIdx;
            detectionIdx = assignments(assignmentsIdx,2);
            
            detectedMarkersForCurrentTrack = detections(detectionIdx, :);
            
            %if size(detectedMarkersForCurrentTrack, 1) > 2 && t > 10
            %   dist = distanceKalman(tracks(currentTrackIdx).kalmanFilter, detectedMarkersForCurrentTrack);
            %   minDist = min(dist, [], 1);
            %   %distToCenter =  sqrt(sum((detectedMarkersForCurrentTrack - tracks(currentTrackIdx).kalmanFilter.x(1:dim)').^2,2));
            %   isValidDetections = minDist < 30;
            %   detectedMarkersForCurrentTrack = detectedMarkersForCurrentTrack(isValidDetections', :);
            %end
            
            % Correct the estimate of the object's location
            % using the new detection.
            tracks(currentTrackIdx).kalmanFilter.z = reshape(detectedMarkersForCurrentTrack, [], 1);
            tracks(currentTrackIdx).kalmanFilter = correctKalman(tracks(currentTrackIdx).kalmanFilter, 1, tracks(currentTrackIdx).kalmanParams);
            
            % Replace predicted bounding box with detected
            % bounding box.
            
            %TODO should be contained in klamanFilter object
            %tracks(trackIdx).center = getCenter(tracks(i).pattern, detectedMarkersForCurrentTrack, tracks(i).kalmanFilter);
            %tracks(trackIdx).markers = detectedMarkersForCurrentTrack;
            
            % Update track's age.
            tracks(currentTrackIdx).age = tracks(currentTrackIdx).age + 1;
            
            % Update visibility.
            tracks(currentTrackIdx).totalVisibleCount = tracks(currentTrackIdx).totalVisibleCount + 1;
            tracks(currentTrackIdx).consecutiveInvisibleCount = 0;
        end
    end

%% Update Unassigned Tracks
% Mark each unassigned track as invisible, and increase its age by 1.

    function updateUnassignedTracks()
        unassignedTracks = double(unassignedTracks);
        allUnassignedTracksIdx = unique(floor((unassignedTracks-1)/nMarkers) + 1);
        nUnassignedTracks = length(allUnassignedTracksIdx);
        for i = 1:nUnassignedTracks
            unassignedTrackIdx = allUnassignedTracksIdx(i);
            if tracks(unassignedTrackIdx).age > 0
                tracks(unassignedTrackIdx).age = tracks(unassignedTrackIdx).age + 1;
                tracks(unassignedTrackIdx).consecutiveInvisibleCount = tracks(unassignedTrackIdx).consecutiveInvisibleCount + 1;
            end
        end
    end

%% Delete Lost Tracks
% The |deleteLostTracks| function deletes tracks that have been invisible
% for too many consecutive frames. It also deletes recently created tracks
% that have been invisible for too many frames overall.

    function deleteLostTracks()
        if isempty(tracks)
            return;
        end
        
        invisibleForTooLong = 150;
        ageThreshold = 0;
        visibilityFraction = 0.5;
        
        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        
        % Find the indices of 'lost' tracks.
        lostIdxBool = (( ages < ageThreshold & visibility < visibilityFraction) | [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong) & (ages > 0);
        lostIdx = find(lostIdxBool);
        if ~isempty(lostIdx)
            for i=1:length(lostIdx)
                %unassignedPatterns{end+1} = tracks(lostIdx(i)).kalmanFilter.pattern;
                unassignedPatterns(lostIdx(i)) = 1;
                % Mark track as lost, i.e. set age to 0
                % TODO: Also wipe other attributes
                tracks(lostIdx(i)).age = 0;
            end
        end
    end




%% Create New Tracks
% Create new tracks from unassigned detections. Assume that any unassigned
% detection is a start of a new track. In practice, you can use other cues
% to eliminate noisy detections, such as size, location, or appearance.

    function createNewTracks()
        if length(unassignedDetections) > 1 && sum(unassignedPatterns) > 0
            epsilon = 100;
            clustersRaw = clusterUnassignedDetections(detections(unassignedDetections,:), epsilon);
            nClusters = 0;
            for i=1:length(clustersRaw)
                if size(clustersRaw{i},1) == 4
                    clusters{i} = clustersRaw{i};
                    nClusters = nClusters + 1;
                elseif size(clustersRaw{i},1) == 3
                    centers{i} = mean(clustersRaw{i}, 1);
                    nClusters = nClusters + 1;
                end
            end
            
            if nClusters < 1
                return
            end
            if isempty(centers)
                costMatrix = zeros(sum(unassignedPatterns), length(clusters));
                rotMatsMatrix = zeros(sum(unassignedPatterns), length(clusters), 3,3);
                translationsMatrix = zeros(sum(unassignedPatterns), length(clusters), 3);
                unassignedPatternsIdx = find(unassignedPatterns);
                for i = 1:sum(unassignedPatterns)
                    for j = 1:length(clusters)
                        pattern = squeeze(patterns(unassignedPatternsIdx(i),:,:));
                        % TODO bettern method that works for different number
                        % of points as well
                        [R, translation, MSE] = umeyama(pattern', detections(clusters{j},:)');
                        costMatrix(i,j) = MSE;
                        rotMatsMatrix(i,j,:,:) = R;
                        translationsMatrix(i,j,:) = translation;
                    end
                end
                
                costOfNonAssignment = 3000;
                [patternToClusterAssignment, stillUnassignedPatterns, ~] = ...
                    assignDetectionsToTracks(costMatrix, costOfNonAssignment);
                
                
                %for each (i,j) in patternToClusterAssignment createNewTrack
                for i=1:size(patternToClusterAssignment,1)
                    specificPatternIdx = patternToClusterAssignment(i,1);
                    clusterIdx = patternToClusterAssignment(i,2);
                    pos = squeeze( translationsMatrix(specificPatternIdx, clusterIdx,:) );
                    quat = matToQuat( squeeze(rotMatsMatrix(specificPatternIdx, clusterIdx, :,:)) );
                    
                    % Create a Kalman filter object.
                    %TODO adaptive initial Noise!!!!
                    patternIdx = unassignedPatternIdx(specificPatternIdx);
                    pattern = squeeze( patterns(patternIdx,:,:));
                    [s, kalmanParams] = setupKalman(pattern, -1, model, quatMotionType, measurementNoise, processNoise, initialNoise);
                    if strcmp(quatMotionType, 'brownian')
                        s.x = [pos'; zeros(3,1); quat'];
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
            %TODO WTF is this case distinction?
            % TODO add method to initialize 3 clustered unassigned
            % detections, i.e. with match_patterns()
            else
                    patternIdx = find(unassignedPatterns);
                    pattern = squeeze( patterns(patternIdx,:,:));
                    [s, kalmanParams] = setupKalman(pattern, -1, model, quatMotionType, measurementNoise, processNoise, initialNoise);
                    if strcmp(quatMotionType, 'brownian')
                        s.x = [centers{1}'; zeros(3,1); [sqrt(2);0;0;0]];
                        % TODO also estimate uncertainty
                        s.P = eye(2*dim+4) .* repelem([kalmanParams.initPositionVar; kalmanParams.initMotionVar; kalmanParams.initQuatVar], [dim, dim, 4]);
                    else
                        s.x = [centers{1}'; zeros(3,1); [sqrt(2);0;0;0]; zeros(4,1)];
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
            
            
            
            % then do pattern matching
            % then maybe umeyama to get rotation an translation
            % from that get initial center and rotation
            % Question how many detections in a cluster do I need for it to
            % work?
            
        end
    end




%% Vizualization methods

    % This function sets up the figure.
    %
    function initializeFigure()
        figure;
        %scatter3([minPos(1), maxPos(1)], [minPos(2), maxPos(2)], [minPos(3), maxPos(3)], '*')
        hold on;
        if shouldShowTruth && exist('trueTrajectory', 'var')
            for k = 1:nObjects
                trueTrajectories{k} = plot3(trueTrajectory(k,1,1),trueTrajectory(k,1,2), trueTrajectory(k,1,3), 'Color', colorsTrue(k,:));
            end
        end
        
        for k = 1:nObjects
            birdsTrajectories{k} = plot3(NaN, NaN, NaN, 'Color', colorsPredicted(k,:));
        end
        
        for k = 1:nObjects
            dets = squeeze(D(1,(k-1)*nMarkers+1:k*nMarkers,:));
            markersForVisualization{k} = plot3(dets(:,1),dets(:,2), dets(:,3), '*', 'MarkerSize', 5, 'MarkerEdgeColor', [0.5; 0.5; 0.5]);
            %birdsPositions{k} = plot3(NaN, NaN, NaN, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', colors(k,:));
            for n = 1:nMarkers
               markerPositions{k,n} = plot3(NaN, NaN, NaN, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', colorsPredicted(k,:));
               viconMarkerPositions{k,n} = plot3(NaN, NaN, NaN, 'square', 'MarkerSize', 12, 'MarkerEdgeColor', colorsTrue(k,:));
            end
        end
        
        
        grid on;
        axis equal;
        axis manual;
    end

    function displayTrackingResults()
        for k = 1:nObjects
            if t < T && t > 1
                if shouldShowTruth && exist('trueTrajectory', 'var') && size(trueTrajectory,2) > t
                    newXTrue = [trueTrajectories{k}.XData trueTrajectory(k,t,1)];
                    newYTrue = [trueTrajectories{k}.YData trueTrajectory(k,t,2)];
                    newZTrue = [trueTrajectories{k}.ZData trueTrajectory(k,t,3)];
                    
                    vizLength = length(newXTrue);
                    if ~keepOldTrajectory && vizLength > vizHistoryLength
                       newXTrue = newXTrue(1,vizLength-vizHistoryLength:vizLength);
                       newYTrue = newYTrue(1,vizLength-vizHistoryLength:vizLength);
                       newZTrue = newZTrue(1,vizLength-vizHistoryLength:vizLength);
                    end
                    
                    trueTrajectories{k}.XData = newXTrue;
                    trueTrajectories{k}.YData = newYTrue;
                    trueTrajectories{k}.ZData = newZTrue; 
                    
                    pattern = tracks(k).kalmanFilter.pattern;
                    trueRotMat = Rot(trueOrientation(k, t, :));
                    trueRotatedPattern = (trueRotMat * pattern')';
                    
                    for n = 1:nMarkers
                        viconMarkerPositions{k,n}.XData = trueTrajectory(k, t, 1) + trueRotatedPattern(n,1);
                        viconMarkerPositions{k,n}.YData = trueTrajectory(k, t, 2) + trueRotatedPattern(n,2);
                        viconMarkerPositions{k,n}.ZData = trueTrajectory(k, t, 3) + trueRotatedPattern(n,3);
                    end
                end
                
                if tracks(k).age > 0
                    xPos = tracks(k).kalmanFilter.x(1);
                    yPos = tracks(k).kalmanFilter.x(2);
                    zPos = tracks(k).kalmanFilter.x(3);

                    newXData = [birdsTrajectories{k}.XData xPos];
                    newYData = [birdsTrajectories{k}.YData yPos];
                    newZData = [birdsTrajectories{k}.ZData zPos];
                    % only plot the trajectory in the most recent 1000
                    % frames.
                    vizLength = length(newXData);
                    if ~keepOldTrajectory && vizLength > vizHistoryLength
                       newXData = newXData(1,vizLength-vizHistoryLength:vizLength);
                       newYData = newYData(1,vizLength-vizHistoryLength:vizLength);
                       newZData = newZData(1,vizLength-vizHistoryLength:vizLength);
                    end
                    birdsTrajectories{k}.XData = newXData;
                    birdsTrajectories{k}.YData = newYData;
                    birdsTrajectories{k}.ZData = newZData;
                
                    %birdsPositions{k}.XData = xPos;
                    %birdsPositions{k}.YData = yPos;
                    %birdsPositions{k}.ZData = zPos;
                    
                    pattern = tracks(k).kalmanFilter.pattern;
                    quat = tracks(k).kalmanFilter.x(7:10);
                    rotMat = Rot(quat);
                    rotatedPattern = (rotMat * pattern')';
                    
                    for n = 1:nMarkers
                        markerPositions{k,n}.XData = xPos + rotatedPattern(n,1);
                        markerPositions{k,n}.YData = yPos + rotatedPattern(n,2);
                        markerPositions{k,n}.ZData = zPos + rotatedPattern(n,3);
                    end
                end
            end
            dets = squeeze(D(t,(k-1)*nMarkers+1:k*nMarkers,:));
            markersForVisualization{k}.XData = dets(:,1);
            markersForVisualization{k}.YData = dets(:,2);
            markersForVisualization{k}.ZData = dets(:,3);
        end
        drawnow
    end


%% helper functions
    function cluster_centers = process_clusters(clusters)
        %num_clusters = sum(~cellfun(@isempty,clusters),2);
        num_clusters = 0;
        for k = 1:length(clusters)
            if size(clusters{1,k},1) > 1
                num_clusters = num_clusters + 1;
            end
        end
        
        %TODO viellicht cluster mit size 1 wegnehmen
        %TODO Checken ob vielleicht sogar zu fein geclustert wird, das könnte in geschossen reultieren
        
        cluster_centers = zeros(num_clusters,3);
        idx = 1;
        for c = 1:length(clusters)
            if size(clusters{1,c},1) > 1
                cluster_centers(idx,:) = mean(clusters{1,c},1);
                idx = idx + 1;
            end
        end
    end

    function all_detections = combine_detections(assgn, unassgn)
        is_new_detection = false(size(unassgn));
        for i = 1:size(unassgn,1)
            p = unassgn(i,:);
            d = sqrt(sum((assgn - p).^2,2));
            is_new_detection(i) = min(d) > 230;
        end
        all_detections = [assgn;unassgn(is_new_detection,:)];
    end

end
