%% Motion-Based Multiple Object Tracking
% This example shows how to perform automatic detection and motion-based
% tracking of moving objects in a video from a stationary camera.
%
%   Copyright 2014 The MathWorks, Inc.

%%
% Detection of moving objects and motion-based tracking are important
% components of many computer vision applications, including activity
% recognition, traffic monitoring, and automotive safety.  The problem of
% motion-based object tracking can be divided into two parts:
%
% # Detecting moving objects in each frame
% # Associating the detections corresponding to the same object over time
%
% The detection of moving objects uses a background subtraction algorithm
% based on Gaussian mixture models. Morphological operations are applied to
% the resulting foreground mask to eliminate noise. Finally, blob analysis
% detects groups of connected pixels, which are likely to correspond to
% moving objects.
%
% The association of detections to the same object is based solely on
% motion. The motion of each track is estimated by a Kalman filter. The
% filter is used to predict the track's location in each frame, and
% determine the likelihood of each detection being assigned to each
% track.
%
% Track maintenance becomes an important aspect of this example. In any
% given frame, some detections may be assigned to tracks, while other
% detections and tracks may remain unassigned. The assigned tracks are
% updated using the corresponding detections. The unassigned tracks are
% marked invisible. An unassigned detection begins a new track.
%
% Each track keeps count of the number of consecutive frames, where it
% remained unassigned. If the count exceeds a specified threshold, the
% example assumes that the object left the field of view and it deletes the
% track.
%
% For more information please see
% <matlab:helpview(fullfile(docroot,'toolbox','vision','vision.map'),'multipleObjectTracking') Multiple Object Tracking>.
%
% This example is a function with the main body at the top and helper
% routines in the form of
% <matlab:helpview(fullfile(docroot,'toolbox','matlab','matlab_prog','matlab_prog.map'),'nested_functions') nested functions>
% below.

function [estimatedPositions, estimatedQuats] = ownMOT(D, patterns, initialStates, nObjects, trueTrajectory)

% Create System objects used for reading video, detecting moving objects,
% and displaying the results.

nMarkers = 4;

[T, ~, dim] = size(D);
%visParams.maxPos = squeeze(max(D,[],[1 2]));
%visParams.minPos = squeeze(min(D,[],[1 2]));
maxPos = squeeze(max(D,[],[1 2]));
minPos = squeeze(min(D,[],[1 2]));

%visParams.trueTrajectory = trueTrajectory;
%visParams.D = D;
%visParams.nMarkers = nMarkers;
%visParams.T = T;

processNoise.position = 30;
processNoise.motion = 20;
processNoise.quat = 30;
processNoise.quatMotion = 30;
measurementNoise = 5000;
model = 'extended';
initialNoise.initPositionVar = 10;
initialNoise.initMotionVar = 100;
initialNoise.initQuatVar = 10000;
initialNoise.initQuatMotionVar = 1000;

nextId = 1; % ID of the next track
tracks = initializeTracks();

%P = cell(nObjects ,1);
%visParams.markersForVisualization = cell(nObjects,1);
markersForVisualization = cell(nObjects,1);


%visParams = animateMOT('init', visParams);
initializeFigure();

estimatedPositions = zeros(nObjects, T, 3);
estimatedQuats = zeros(nObjects, T, 4);


% Detect moving objects, and track them across video frames.
for t = 1:T
    detections = squeeze(D(t,:,:));
    detections = reshape(detections(~isnan(detections)),[],dim);
    
    predictNewLocationsOfTracks();
    [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignment();
    
    updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks();
    %TODO implement createNewTracks
    %createNewTracks();
    
    %displayTrackingResults();
    if t > 1 && t < 1000
        %visParams.tracks = tracks;
        %visParams.oldTracks = oldTracks;
        %visParams = animateMOT('step', visParams, t);
        displayTrackingResults();
    end
    
    % Store tracking results
    for ii = 1:nObjects
        state = tracks(ii).kalmanFilter.x;
        estimatedPositions(ii,t,:) = state(1:dim);
        estimatedQuats(ii,t,:) = state(2*dim+1:2*dim+4);
    end
    
    oldTracks = tracks;
end



%% Initialize Tracks
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

    function tracks = initializeTracks()
        for i = 1:nObjects
            [s, kalmanParams] = setupKalman(squeeze(patterns(i,:,:)), -1, model, measurementNoise, processNoise, initialNoise);
            s.x = initialStates(i,:)';
            s.P = eye(2*dim+8) .* repelem([kalmanParams.initPositionVar; kalmanParams.initMotionVar; kalmanParams.initQuatVar; kalmanParams.initQuatMotionVar], [dim, dim, 4, 4]);
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
            % Predict the current location of the track.
            tracks(i).kalmanFilter = predictKalman(tracks(i).kalmanFilter, 1, tracks(i).kalmanParams, 'extended');
        end
    end

%% Assign Detections to Tracks
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

    function [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignment()
        
        nTracks = length(tracks);
        nDetections = size(detections, 1);
        
        % Compute the cost of assigning each detection to each marker.
        cost = zeros(nTracks*nMarkers, nDetections);
        for i = 1:nTracks
            cost((i-1)*nMarkers+1:i*nMarkers, :) = distanceKalman(tracks(i).kalmanFilter, detections);
        end
        
        % Solve the assignment problem.
        costOfNonAssignment = 3000;
        [assignments, unassignedTracks, unassignedDetections] = assignDetectionsToTracks(cost, costOfNonAssignment);
    end


    function [assignments, unassignedTracks, unassignedDetections] = newDetectionToTrackAssignment()
        
        nTracks = length(tracks);
        nDetections = size(detections, 1);
        
        % Compute the cost of assigning each detection to each marker.
        cost = zeros(nTracks*nMarkers, nDetections);
        for i = 1:nTracks
            cost((i-1)*nMarkers+1:i*nMarkers, :) = distanceKalman(tracks(i).kalmanFilter, detections);
        end
        
        % Solve the assignment problem.
        costOfNonAssignment = 100;
        [assignments, unassignedTracks, unassignedDetections] = assignDetectionsToTracks(cost, costOfNonAssignment);
    end

%% Update Assigned Tracks
% The |updateAssignedTracks| function updates each assigned track with the
% corresponding detection. It calls the |correct| method of
% |vision.KalmanFilter| to correct the location estimate. Next, it stores
% the new bounding box, and increases the age of the track and the total
% visible count by 1. Finally, the function sets the invisible count to 0.

    function updateAssignedTracks()
        assignments = double(assignments);
        allAssignedTracksIdx = unique(floor((assignments(:,1)-1)/nMarkers) + 1);
        nAssignedTracks = length(allAssignedTracksIdx);
        for i = 1:nAssignedTracks
            currentTrackIdx = allAssignedTracksIdx(i);
            assignmentsIdx = floor((assignments(:,1)-1)/nMarkers) + 1 == currentTrackIdx;
            detectionIdx = assignments(assignmentsIdx,2);
            detectedMarkersForCurrentTrack = detections(detectionIdx, :);
            
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
            tracks(unassignedTrackIdx).age = tracks(unassignedTrackIdx).age + 1;
            tracks(unassignedTrackIdx).consecutiveInvisibleCount = tracks(unassignedTrackIdx).consecutiveInvisibleCount + 1;
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
        ageThreshold = 5;
        visibilityFraction = 0.5;
        
        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        
        % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < visibilityFraction) | [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
        
        % Delete lost tracks.
        tracks = tracks(~lostInds);
    end




%% Create New Tracks
% Create new tracks from unassigned detections. Assume that any unassigned
% detection is a start of a new track. In practice, you can use other cues
% to eliminate noisy detections, such as size, location, or appearance.

    function createNewTracks()
        if ~isempty(unassignedDetections)

            %TODO
            [positions, quaternions, patternIdx] = match_patterns( detections(unassignedDetections, :), patterns, 'initial');

            for i = 1:size(positions, 1)

                pos = positions(i,:);
                quat = quaternions(i,:);

                % Create a Kalman filter object.
                %kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                %    center, [100, 40], [100, 70], 100);

                [s, kalmanParams] = setupKalman(squeeze(patterns(i,:,:)), -1, model, measurementNoise, processNoise, initialNoise);
                s.x = [pos'; zeros(3,1); quat'; zeros(4,1)];
                % TODO also estimate uncertainty
                s.P = eye(2*dim+8) .* repelem([kalmanParams.initPositionVar; kalmanParams.initMotionVar; kalmanParams.initQuatVar; kalmanParams.initQuatMotionVar], [dim, dim, 4, 4]);
                s.pattern = squeeze(patterns(patternIdx(i),:,:));
                tracks(i) = struct(...
                    'id', nextId, ... %'center', , ...
                    'kalmanFilter', s, ...
                    'kalmanParams', kalmanParams, ...
                    'age', 1, ...
                    'totalVisibleCount', 1, ...
                    'consecutiveInvisibleCount', 0);

                % Add it to the array of tracks.
                tracks(end + 1) = newTrack;

                % Increment the next id.
                nextId = nextId + 1;
            end
        end
    end




%% Display Tracking Results
% The |displayTrackingResults| function draws a bounding box and label ID
% for each track on the video frame and the foreground mask. It then
% displays the frame and the mask in their respective video players.

    function initializeFigure()
        figure;
        colors = distinguishable_colors(nObjects);
        scatter3([minPos(1), maxPos(1)], [minPos(2), maxPos(2)], [minPos(3), maxPos(3)], '*')
        hold on;
        if exist('trueTrajectory', 'var')
            for k = 1:nObjects
                plot3(trueTrajectory(k,1:2,1),trueTrajectory(k,1:2,2), trueTrajectory(k,1:2,3), 'Color', colors(k,:));
            end
        end
        for k = 1:nObjects
            dets = squeeze(D(1,(k-1)*nMarkers+1:k*nMarkers,:));
            markersForVisualization{k} = plot3(dets(:,1),dets(:,2), dets(:,3), '*', 'MarkerSize', 10, 'MarkerEdgeColor', colors(k,:));
        end
        grid on;
        axis manual
    end

    function displayTrackingResults()
        colorsPredicted = distinguishable_colors(nObjects);
        colorsTrue = (colorsPredicted + 2) ./ (max(colorsPredicted,[],2) +2);
        
        for k = 1:nObjects
            if t < T && t > 1
                if exist('trueTrajectory', 'var')
                    plot3(trueTrajectory(k,t:t+1,1), trueTrajectory(k,t:t+1,2), ...
                          trueTrajectory(k,t:t+1,3), 'Color', colorsTrue(k,:));
                end
                plot3( [oldTracks(k).kalmanFilter.x(1); tracks(k).kalmanFilter.x(1)], ...
                       [oldTracks(k).kalmanFilter.x(2); tracks(k).kalmanFilter.x(2)], ...
                       [oldTracks(k).kalmanFilter.x(3); tracks(k).kalmanFilter.x(3)], ... 
                       'Color', colorsPredicted(k,:));
            end
            dets = squeeze(D(t,(k-1)*nMarkers+1:k*nMarkers,:));
            markersForVisualization{k}.XData = dets(:,1);
            markersForVisualization{k}.YData = dets(:,2);
            markersForVisualization{k}.ZData = dets(:,3);
        end
        drawnow
        %pause(0.1)
    end

    

%% Summary
% This example created a motion-based system for detecting and
% tracking multiple moving objects. Try using a different video to see if
% you are able to detect and track objects. Try modifying the parameters
% for the detection, assignment, and deletion steps.
%
% The tracking in this example was solely based on motion with the
% assumption that all objects move in a straight line with constant speed.
% When the motion of an object significantly deviates from this model, the
% example may produce tracking errors. Notice the mistake in tracking the
% person labeled #12, when he is occluded by the tree.
%
% The likelihood of tracking errors can be reduced by using a more complex
% motion model, such as constant acceleration, or by using multiple Kalman
% filters for every object. Also, you can incorporate other cues for
% associating detections over time, such as size, shape, and color.

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
        %TODO Checken ob vielleicht sogar zu fein geclustert wird, das k�nnte in geschossen reultieren
        
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
