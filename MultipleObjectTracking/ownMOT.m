%% Multi Object Tracking
% The original code structure comes from the MATLAB tutorial for motion
% based multi object tracking:
% https://de.mathworks.com/help/vision/examples/motion-based-multiple-object-tracking.html

% TODO
% Explanation goes here

function [estimatedPositions, estimatedQuats, positionVariance, rotationVariance, snapshots] = ownMOT(D, patterns, patternNames, useVICONinit, initialStates, nObjects, shouldShowTruth, trueTrajectory, trueOrientation, quatMotionType, hyperParams)
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

processNoise.position = hyperParams.posNoise;
processNoise.positionBrownian = hyperParams.posNoiseBrownian;
processNoise.motion = hyperParams.motNoise;
processNoise.acceleration = hyperParams.accNoise;
processNoise.quat = hyperParams.quatNoise;
processNoise.quatMotion = hyperParams.quatMotionNoise;
measurementNoise = hyperParams.measurementNoise;
model =  hyperParams.modelType; %'extended'; %'LieGroup'; %
initialNoise.initPositionVar = 5;
initialNoise.initMotionVar = 50;
initialNoise.initAccVar = 50;
initialNoise.initQuatVar = 0.05;
initialNoise.initQuatMotionVar = 0.075;

params.initialNoise = initialNoise;
params.model = model;
params.measurementNoise = measurementNoise;
params.processNoise = processNoise;
params.quatMotionType = quatMotionType;
params.motionType = 'constAcc';

params.minDistToBird = hyperParams.minDistToBird;
params.initThreshold = hyperParams.initThreshold;
params.initThreshold4 = hyperParams.initThreshold4;

params.initMotionModel = 0;


similarPairs = getSimilarPatterns(patterns, hyperParams.patternSimilarityThreshold);


nextId = 1;
if useVICONinit
    unassignedPatterns = zeros(nObjects, 1);
else
    unassignedPatterns = ones(nObjects, 1);
end
[tracks, ghostTracks] = initializeTracks();



markersForVisualization = cell(1,1);
ghostBirdsVis = cell(1,1);
birdsTrajectories = cell(nObjects,1);
trueTrajectories = cell(nObjects,1);
%birdsPositions = cell(nObjects,1);
markerPositions = cell(nObjects, nMarkers);
viconMarkerPositions = cell(nObjects, nMarkers);

colorsPredicted = distinguishable_colors(nObjects);
colorsTrue = (colorsPredicted + 2) ./ (max(colorsPredicted,[],2) +2);
keepOldTrajectory = 0;
visualizeTracking = hyperParams.visualizeTracking;
%shouldShowTruth = 1;
vizHistoryLength = 200;
if visualizeTracking == 1
    initializeFigure();
end


estimatedPositions = zeros(nObjects, T, 3);
estimatedQuats = zeros(nObjects, T, 4);
positionVariance = zeros(nObjects, T);
rotationVariance = zeros(nObjects, T);

snapshots = {};
snapshotIdx = 1;
freshInitis = zeros(nObjects, 1);

for t = 1:T
    freshInits = zeros(nObjects,1);
    %tic
    detections = squeeze(D(t,:,:));
    detections = reshape(detections(~isnan(detections)),[],dim);
    
    predictNewLocationsOfTracks();
    [assignedTracks, unassignedTracks, assignedGhostTracks, unassignedGhostTracks, unassignedDetections] = detectionToTrackAssignment();
    %[assignedTracks, unassignedTracks, assignedGhostTracks, unassignedGhostTracks, unassignedDetections] = detectionToTrackAssignment();

    [deletedGhostTracks, rejectedDetections] = updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks(deletedGhostTracks);
    unusedDets = [detections(unassignedDetections, :); rejectedDetections];

    if sum(unassignedPatterns) > 0 &&  length(unusedDets) > 1
        [tracks, ghostTracks, unassignedPatterns, extraFreshInits] = createNewTracks(unusedDets, unassignedPatterns, tracks, patterns, params, patternNames, similarPairs, ghostTracks);
        freshInits = freshInits | extraFreshInits;
    end
    t
    if t == 686
       t %rot
    end
    %if t==5000
    %   t % total chaos
    %end % 8700grün
    if visualizeTracking == 1
        displayTrackingResults();
    end
    
    % Store tracking results
    for ii = 1:nObjects
        if tracks(ii).age > 0
            if strcmp(model, 'LieGroup')
                p = tracks(ii).kalmanFilter.mu.X(1:3, 4);
                
                estimatedPositions(ii, t, :) = p;
                estimatedQuats(ii, t, :) = rotm2quat(tracks(ii).kalmanFilter.mu.X(1:3,1:3));
                P = tracks(ii).kalmanFilter.P;
                rotationVariance(ii, t) = (P(1,1) + P(2,2) + P(3,3)) / 3;
                positionVariance(ii, t) = (P(4,4) + P(5,5) + P(6,6)) / 3;
            else
                state = tracks(ii).kalmanFilter.x;
                P = tracks(ii).kalmanFilter.P;
                estimatedPositions(ii,t,:) = state(1:dim);
                positionVariance(ii,t) = (P(1,1) + P(2,2) + P(3,3)) / 3;
                if strcmp(params.motionType, 'constAcc')
                    estimatedQuats(ii,t,:) = state(3*dim+1:3*dim+4);
                    rotationVariance(ii,t) = (P(3*dim+1, 3*dim+1) + P(3*dim+2, 3*dim+2) + ... 
                                              P(3*dim+3, 3*dim+3) + P(3*dim+4, 3*dim+4)) / 4;
                else
                    estimatedQuats(ii,t,:) = state(2*dim+1:2*dim+4);
                    rotationVariance(ii,t) = (P(2*dim+1, 2*dim+1) + P(2*dim+2, 2*dim+2) + ...
                                              P(2*dim+3, 2*dim+3) + P(2*dim+4, 2*dim+4)) / 4;
                end
            end
        else
            estimatedPositions(ii,t,:) = ones(3,1) * NaN;
            estimatedQuats(ii,t,:) = ones(4,1) * NaN;
            rotationVariance(ii,t) = NaN;
            positionVariance(ii,t) = NaN;
        end
    end
    
    if any(freshInits)
       state.tracks = tracks;
       state.ghostTracks = ghostTracks;
       state.t = t;
       state.freshInits = find(freshInits);
       snapshots{snapshotIdx} = state;
       snapshotIdx = snapshotIdx + 1;
    end
    %toc
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

    function [tracks, ghostTracks] = initializeTracks()
        if useVICONinit
            for i = 1:nObjects
                [s, kalmanParams] = setupKalman(squeeze(patterns(i,:,:)), -1, params);
                if strcmp(model, 'LieGroup')
                    mu.X = [ quat2rotm(initialStates.quat(i,:)) initialStates.pos(i,:)'; [0 0 0 1] ];
                    mu.v = initialStates.velocity(i,:)';
                    mu.a = initialStates.acceleration(i,:)';
                    s.mu = mu;
                    s.P = diag(repelem([params.initialNoise.initQuatVar; 
                                        params.initialNoise.initPositionVar; 
                                        params.initialNoise.initMotionVar; 
                                        params.initialNoise.initAccVar
                                       ],[dim, dim, dim, dim]));
                else
                    %if strcmp(quatMotionType, 'brownian')
                    %    s.x = [initialStates.pos(i,:) initialStates.velocity(i,:) initialStates.quat(i,:)]'; %;initialStates(i,1:end-4)';
                    %    s.P = eye(2*dim+4) .* repelem([kalmanParams.initPositionVar; kalmanParams.initMotionVar; kalmanParams.initQuatVar], [dim, dim, 4]);
               
                    if strcmp(params.motionType, 'constAcc')
                        %TODO: add acceleration=0 in initial state
                        s.x = [initialStates.pos(i,:) initialStates.velocity(i,:) initialStates.acceleration(i,:) initialStates.quat(i,:)]'; %initialStates(i,1:end-4);
                        s.P = eye(3*dim+4) .* repelem([kalmanParams.initPositionVar; kalmanParams.initMotionVar; kalmanParams.initAccVar; kalmanParams.initQuatVar], [dim, dim, dim, 4]);
                    else
                        s.x = [initialStates.pos(i,:) initialStates.velocity(i,:) initialStates.acceleration(i,:) initialStates.quat(i,:) 0 0 0 0]';%initialStates(i,:)';
                        s.P = eye(2*dim+8) .* repelem([kalmanParams.initPositionVar; kalmanParams.initMotionVar; kalmanParams.initQuatVar; kalmanParams.initQuatMotionVar], [dim, dim, 4, 4]);
                    end
                end
                s.pattern = squeeze(patterns(i,:,:));
                s.flying = -1;
                s.consecutiveInvisibleCount = 0;
                tracks(i) = struct(...
                    'id', i, ... 
                    'name', patternNames{i}, ...
                    'kalmanFilter', s, ...
                    'kalmanParams', kalmanParams, ...
                    'age', 1, ...
                    'totalVisibleCount', 1, ...
                    'consecutiveInvisibleCount', 0);
            end
        else
            emptyTrack.age = 0;
            emptyTrack.totalVisibleCount = 0;
            emptyTrack.consecutiveInvisibleCount = 0;
            
            for in =1:nObjects
                emptyTrack.id = in;
                emptyTrack.name = '';
                emptyTrack.kalmanParams = [];
                kalmanFilter.pattern = squeeze(patterns(in,:,:));
                kalmanFilter.lastSeen = NaN*zeros(3,1);
                kalmanFilter.lastVisibleFrame = NaN;
                emptyTrack.kalmanFilter = kalmanFilter;
                tracks(in) =  emptyTrack;
            end
            unassignedDetections = squeeze(D(1,:,:));
            unassignedDetections = reshape(unassignedDetections(~isnan(unassignedDetections)),[],dim);
            [tracks, ghostTracks, unassignedPatterns, freshInits] = createNewTracks(unassignedDetections, unassignedPatterns, tracks, patterns, params, patternNames, similarPairs);
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
        
        for i = 1:length(ghostTracks)
           if ghostTracks(i).age > 0
              ghostKF = ghostTracks(i).kalmanFilter;
              ghostKF.x = ghostKF.F * ghostKF.x;
              ghostKF.P = ghostKF.F * ghostKF.P * ghostKF.F' + ghostKF.Q;
              ghostTracks(i).kalmanFilter = ghostKF;
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

    function [assignedTracks, unassignedTracks, assignedGhostTracks, unassignedGhostTracks, unassignedDetections] = detectionToTrackAssignment()
        
        nTracks = length(tracks)+length(ghostTracks);
        nDetections = size(detections, 1);
        
        % Compute the cost of assigning each detection to each marker.
        cost = zeros(nTracks*nMarkers, nDetections);
        for i = 1:nTracks
            if i <= length(tracks)
                if tracks(i).age > 0
                    %TODO: something more sophisticated here would be useful!
                    %TODO: bc. costOfNonAssignment can mess up
                    cost((i-1)*nMarkers+1:i*nMarkers, :) = distanceKalman(tracks(i).kalmanFilter, detections, params.motionType);
                else
                    cost((i-1)*nMarkers+1:i*nMarkers, :) = Inf;
                end
            else
                %TODO allow max of nMarkers markers to be assigned to
                %ghost bird??
                if ghostTracks(i-length(tracks)).age > 0
                    cost((i-1)*nMarkers+1:i*nMarkers, :) = 1.3*repmat(...
                        pdist2( ghostTracks(i-length(tracks)).kalmanFilter.x(1:3)', detections), ...
                                                                  [nMarkers, 1]);
                else
                   cost((i-1)*nMarkers+1:i*nMarkers, :) = Inf; 
                end
                
            end
        end
        
        % Solve the assignment problem.
        costOfNonAssignment = hyperParams.costOfNonAsDtTA;
        [assignments, unassignments, unassignedDetections] = assignDetectionsToTracks(cost, costOfNonAssignment);
        % Partition results into tracks and ghost tracks correspondingly
        assignedTracks = assignments(assignments(:, 1) <= length(tracks)*nMarkers, :);
        assignedGhostTracks = assignments(assignments(:, 1) > length(tracks)*nMarkers, :);
        unassignedTracks = unassignments(unassignments <= length(tracks)*nMarkers);
        unassignedGhostTracks = unassignments(unassignments > length(tracks)*nMarkers);
    end

function [assignedTracks, unassignedTracks, assignedGhosts, unassignedGhosts, unassignedDetections] = detectionToTrackAssignmentNew()
        
        nTracks = length(tracks)+length(ghostTracks);
        nDetections = size(detections, 1);
        
        trackAssignments = cell(length(tracks),2);
        ghostAssignments = cell(length(ghostTracks),2);
        nAssignedTracks = 0;
        nAssignedGhosts = 0;
        unassignedDetections = zeros(0,1);
        for d=1:nDetections
            costs = zeros(nTracks,1);
            for i=1:nTracks
                if i <=length(tracks)
                    if tracks(i).age > 0
                        expectedMarkerLocations = measFuncNonHomogenous(tracks(i).kalmanFilter.mu, tracks(i).kalmanFilter.pattern);
                        costs(i) = min(pdist2(detections(d,:), expectedMarkerLocations));
                    else
                        costs(i) = Inf;
                    end
                else
                    costs(i) = norm(detections(d,:)'-ghostTracks(i-length(tracks)).kalmanFilter.x(1:3));
                end
            end
            [minCost, minIdx] = min(costs);
            if minCost < 120
                if minIdx <= length(tracks)
                    trackAssignments{minIdx, 1} = [trackAssignments{minIdx, 1};
                                                detections(d,:)];
                    trackAssignments{minIdx, 2} = [trackAssignments{minIdx, 2};
                                                   d];
                    if size(trackAssignments{minIdx, 2}, 1) <= 4
                        nAssignedTracks = nAssignedTracks + 1;
                    end

                else
                    ghostAssignments{minIdx-length(tracks),1} = [ghostAssignments{minIdx-length(tracks), 1};
                                                               detections(d,:)];
                    ghostAssignments{minIdx-length(tracks),2} = [ghostAssignments{minIdx-length(tracks), 2};
                                                                 d];
                    if size(ghostAssignments{minIdx-length(tracks), 2}, 1) <= 4
                        nAssignedGhosts = nAssignedGhosts + 1;
                    end
                end
            else
                unassignedDetections(end+1, 1) = d;
            end
            
        end
        
        assignedTracks = zeros(nAssignedTracks, 2);
        unassignedTracks = zeros(length(tracks)*nMarkers-nAssignedTracks, 1);
        
        assignedTrackIdx = 1;
        unassignedTrackIdx = 1;
        for i=1:size(trackAssignments,1)
            dets = trackAssignments{i, 1};
            nDets = size(dets,1);
            detsIdx = trackAssignments{i, 2};
            % If more than 4 dets are nearby a bird, use HA to select best
            % 4 dets
            if nDets > 4
                expectedMarkerLocations = measFuncNonHomogenous(tracks(i).kalmanFilter.mu, tracks(i).kalmanFilter.pattern);
                asg = munkers(pdist2(expectedMarkerLocations, dets));
                detsIdx = detsIdx(asg');
                nDets = 4;
            end
               assignedTracks(assignedTrackIdx:assignedTrackIdx+nDets-1, 1) = i*nMarkers;
               assignedTracks(assignedTrackIdx:assignedTrackIdx+nDets-1, 2) = detsIdx;
               assignedTrackIdx = assignedTrackIdx + nDets;
               unassignedTracks(unassignedTrackIdx:unassignedTrackIdx+4-nDets-1) = i*nMarkers;
               unassignedTrackIdx = unassignedTrackIdx + 4 - nDets;            
            
        end
        assignedGhosts = zeros(nAssignedGhosts, 1);
        unassignedGhosts = zeros(length(ghostTracks)*nMarkers-nAssignedGhosts, 1);
        assignedGhostsIdx = 1;
        unassignedGhostsIdx = 1;
        for i=1:size(ghostAssignments,1)
            dets = ghostAssignments{i, 1};
            nDets = size(dets,1);
            detsIdx = ghostAssignments{i, 2};
            % If more than 4 dets are nearby a bird, use HA to select best
            % 4 dets
            if nDets > 4
                [~, asg] = mink(pdist2(ghostTracks(i).kalmanFilter.x(1:3)', dets), 4);
                detsIdx = detsIdx(asg');
                nDets = 4;
            end
               assignedGhosts(assignedGhostsIdx:assignedGhostsIdx+nDets-1, 1) = i*nMarkers+length(tracks)*nMarkers;
               assignedGhosts(assignedGhostsIdx:assignedGhostsIdx+nDets-1, 2) = detsIdx;
               assignedGhostsIdx = assignedGhostsIdx + nDets;
               unassignedGhosts(unassignedGhostsIdx:unassignedGhostsIdx+4-nDets-1) = i*nMarkers+length(tracks)*nMarkers;
               unassignedGhostsIdx = unassignedGhostsIdx + 4 - nDets;            
        end
    end


%% Update Assigned Tracks
%{
% The |updateAssignedTracks| function updates each assigned track with the
% corresponding detection. It calls the |correct| method of
% |vision.KalmanFilter| to correct the location estimate. Next, it stores
% the new bounding box, and increases the age of the track and the total
% visible count by 1. Finally, the function sets the invisible count to 0.
%}
    function [deletedGhostTracks, allRejectedDetections] = updateAssignedTracks()
        
        allRejectedDetections = zeros(0, 3);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Update tracks.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        assignedTracks = double(assignedTracks);
        allAssignedTracksIdx = unique(floor((assignedTracks(:,1)-1)/nMarkers) + 1);
        nAssignedTracks = length(allAssignedTracksIdx);
        for i = 1:nAssignedTracks
            currentTrackIdx = allAssignedTracksIdx(i);
            assignmentsIdx = floor((assignedTracks(:,1)-1)/nMarkers) + 1 == currentTrackIdx;
            detectionIdx = assignedTracks(assignmentsIdx,2);
            
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
            s = tracks(currentTrackIdx).kalmanFilter;
            if strcmp(model, 'LieGroup')
                s.z = reshape(detectedMarkersForCurrentTrack', [], 1);
                [tracks(currentTrackIdx).kalmanFilter, rejectedDetections] = correctKalman(s, 1, tracks(currentTrackIdx).kalmanParams, 0, hyperParams, tracks(currentTrackIdx).age, params.motionType, currentTrackIdx);
                allRejectedDetections(end + 1: end + size(rejectedDetections, 1), :) = rejectedDetections;
                if s.mu.motionModel > 0 && norm( s.mu.v ) > 35
                    tracks(currentTrackIdx).kalmanFilter.flying = min(s.flying + 2, 10);
                elseif s.mu.motionModel > 0 && norm( s.mu.v ) > 22.5
                    tracks(currentTrackIdx).kalmanFilter.flying = min(s.flying + 1, 10);
                elseif s.mu.motionModel > 0 && norm( s.mu.v ) < 10
                    tracks(currentTrackIdx).kalmanFilter.flying = max(-1, s.flying -2);
                end
            else
                s.z = reshape(detectedMarkersForCurrentTrack, [], 1);
                tracks(currentTrackIdx).kalmanFilter = correctKalman(s, 1, tracks(currentTrackIdx).kalmanParams, 0, hyperParams, tracks(currentTrackIdx).age, params.motionType);
                error('flying indication not implemented for quaternion version')
            end
            
            % Update track's age.
            tracks(currentTrackIdx).age = tracks(currentTrackIdx).age + 1;
            
            % Book keeping: Update visibility, latest positions, an other
            % 'statistic'
            tracks(currentTrackIdx).totalVisibleCount = tracks(currentTrackIdx).totalVisibleCount + 1;
            tracks(currentTrackIdx).consecutiveInvisibleCount = tracks(currentTrackIdx).kalmanFilter.consecutiveInvisibleCount;
            tracks(currentTrackIdx).kalmanFilter.framesInNewMotionModel = tracks(currentTrackIdx).kalmanFilter.framesInNewMotionModel + 1;
            tracks(currentTrackIdx).kalmanFilter.latest5pos(tracks(currentTrackIdx).kalmanFilter.latestPosIdx+1, :) = ...
                tracks(currentTrackIdx).kalmanFilter.mu.X(1:3, 4);
            tracks(currentTrackIdx).kalmanFilter.latestPosIdx = mod(tracks(currentTrackIdx).kalmanFilter.latestPosIdx + 1, 5);
            tracks(currentTrackIdx).kalmanFilter.lastSeen = tracks(currentTrackIdx).kalmanFilter.mu.X(1:3, 4);
            tracks(currentTrackIdx).kalmanFilter.lastVisibleFrame = t;
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Try to identify ghost tracks with pattern.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        assignedGhostTracks(:, 1) = assignedGhostTracks(:, 1) - length(tracks)*nMarkers;
        assignedGhostTracks = double(assignedGhostTracks);
        allAssignedGhostTracksIdx = unique(floor((assignedGhostTracks(:,1)-1)/nMarkers) + 1);
        nAssignedGhostTracks = length(allAssignedGhostTracksIdx);
        deletedGhostTracks = zeros(length(ghostTracks),1);
        
        
        % Unassigned patterns that are not similar to any other unassigned
        % pattern can be used to safely initialize a new track
        % Determine these patterns.
        safePatternsBool = zeros(length(patterns), 1);
        potentialReInit = unassignedPatterns | ([tracks(:).consecutiveInvisibleCount] > 10)';
        assignedPatternsIdx = find(~potentialReInit);
        unassignedPatternsIdx = find(potentialReInit);
        for jj=1:length(unassignedPatternsIdx)
            p = unassignedPatternsIdx(jj);
            conflicts = similarPairs(similarPairs(:, 1) == p, 2);
            conflicts = [conflicts; similarPairs(similarPairs(:, 2) == p, 1)];
            conflicts = setdiff(conflicts, assignedPatternsIdx);
            if isempty(conflicts)
                safePatternsBool(p) = 1;
            end
        end
        
        
        for i = 1:nAssignedGhostTracks
            currentGhostTrackIdx = allAssignedGhostTracksIdx(i);
            assignmentsIdx = floor((assignedGhostTracks(:,1)-1)/nMarkers) + 1 == currentGhostTrackIdx;
            detectionIdx = assignedGhostTracks(assignmentsIdx,2);
            detectedMarkersForCurrentGhostTrack = detections(detectionIdx, :);
            nAssgnDets = size(detectedMarkersForCurrentGhostTrack, 1);
            
            % if  4 detections assigned: run pattern matching and
            %   umeyama, if good fit, init real track!
            % if 2 detections assigned: only run pattern_matching if all
            % simialr patterns are already assigned, in order to avoid
            % id-switches
            if nAssgnDets >= 3 %&& ~(sum(potentialReInit) == 1 + sum(abs(patterns)>=1000, 'all'))
                %if nAssgnDets == 3
                %    unassignedandSafeIdx = find(unassignedPatterns & safePatternsBool);
                %else
                unassignedandSafeIdx = find(potentialReInit);
                %end
                matchingCosts = zeros(length(unassignedandSafeIdx), 1);
                rotations = zeros(length(unassignedandSafeIdx), 3, 3);
                translations = zeros(length(unassignedandSafeIdx), 3);
                for j=1:length(unassignedandSafeIdx)
                   pattern = squeeze(patterns(unassignedandSafeIdx(j), :, :));
                   p = match_patterns(pattern, detectedMarkersForCurrentGhostTrack, 'noKnowledge');
                   assignment = zeros(4,1);
                   assignment(p) = 1:length(p);
                   assignment = assignment(1:size(detectedMarkersForCurrentGhostTrack,1));
                   pattern = pattern(assignment,:);
                   pattern = pattern(assignment > 0, :);
                   [R, translation, MSE] = umeyama(pattern', detectedMarkersForCurrentGhostTrack');
                   matchingCosts(j) = MSE;
                   rotations(j, :, :) = R;
                   translations(j, :) = translation;
                end

                [minCost2, minIdx2] = mink(matchingCosts, 2);
                costDiff = minCost2(2)-minCost2(1);
                patternIdx = unassignedandSafeIdx(minIdx2(1));
                if ~isempty(minCost2(1)) && ...
                    (  ( minCost2(1) < params.initThreshold4 && nAssgnDets == 4 && costDiff > 1)  || ...
                        ( minCost2(1) < params.initThreshold && nAssgnDets == 3 && safePatternsBool(patternIdx)==1 && costDiff > 2) ||... 
                        ( minCost2(1) < 0.2 && nAssgnDets == 3 && costDiff > 2) ...
                    )
                    if nAssgnDets == 3
                        minCost2(1)
                    end
                    if patternIdx == 1 || patternIdx == 2 || patternIdx == 4
                       patternIdx 
                    end
                    pattern = squeeze(patterns(patternIdx, :, :));
                    if isfield(tracks(patternIdx).kalmanFilter, 'mu')
                        newTrack = createLGEKFtrack(squeeze(rotations(minIdx2(1), :, :)), ...
                                    squeeze(translations(minIdx2(1), :))', MSE, patternIdx, ...
                                    pattern, patternNames{patternIdx}, params, ...
                                    tracks(patternIdx).kalmanFilter.mu.motionModel, ...
                                    ghostTracks(currentGhostTrackIdx).kalmanFilter);
                    else
                        newTrack = createLGEKFtrack(squeeze(rotations(minIdx2(1), :, :)), ...
                                squeeze(translations(minIdx2(1), :))', MSE, ...
                                patternIdx, pattern, patternNames{patternIdx}, ...
                                params, -1, ...
                                ghostTracks(currentGhostTrackIdx).kalmanFilter);
                    end
                    tracks(patternIdx) = newTrack;
                    unassignedPatterns(patternIdx) = 0;
                    potentialReInit(patternIdx) = 0;
                    freshInits(patternIdx) = true;
                    % mark ghosst bird as deleted and delete after loop
                    deletedGhostTracks(currentGhostTrackIdx) = 1;
                    % continue loop, as we don't have to update position of
                    % ghost bird
                    continue;

                end
                %TODO fix to regular case!!!!!!!!!!
%             elseif ( sum(potentialReInit) == 1 + sum(abs(patterns)>=1000, 'all') && ...
%                          ghostTracks(currentGhostTrackIdx).trustworthyness > hyperParams.minTrustworthyness )%if all but 1 bird are intialized we can initialize even with 2 or less dets
%                 for j=1:size(potentialReInit)
%                     if ( potentialReInit(j) == 1 && ...
%                          ~any(abs(patterns(j, :, :)) >= 1000, 'all') && ...
%                          ~(abs(t-tracks(j).kalmanFilter.lastVisibleFrame) < 15 && ...
%                            norm(tracks(j).kalmanFilter.lastSeen - mean(detectedMarkersForCurrentGhostTrack, 1)') > 150)...
%                      )
%                         pattern = squeeze(patterns(j, :, :));
%                         if isfield(tracks(j).kalmanFilter, 'mu')
%                             newTrack = createLGEKFtrack(eye(3), ...
%                                         mean(detectedMarkersForCurrentGhostTrack, 1)', 5, j, ...
%                                         pattern, patternNames{j}, ...
%                                         params, tracks(j).kalmanFilter.mu.motionModel, ...
%                                         ghostTracks(currentGhostTrackIdx).kalmanFilter);
%                         else
%                             newTrack = createLGEKFtrack(eye(3), ...
%                                         mean(detectedMarkersForCurrentGhostTrack, 1)', 5, j, ...
%                                         pattern, patternNames{j}, ...
%                                         params, -1, ...
%                                         ghostTracks(currentGhostTrackIdx).kalmanFilter);
%                         end
%                         tracks(j) = newTrack;
%                         unassignedPatterns(j) = 0;
%                         potentialReInit(j) = 0;
%                         % mark ghosst bird as deleted and delete after loop
%                         deletedGhostTracks(currentGhostTrackIdx) = 1;
%                         % continue loop, as we don't have to update position of
%                         % ghost bird
%                         break;
%                     end
%                 end
%                 continue;
            end
             
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % update remaining ghost tracks
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             maxDistToGhost = hyperParams.ghostFPFilterDist;
                         distToGhost = pdist2(ghostTracks(currentGhostTrackIdx).kalmanFilter.x(1:3)', ...
                                  detectedMarkersForCurrentGhostTrack);      
             allRejectedDetections(end + 1: end + nnz(distToGhost > maxDistToGhost), :) = ...
                 detectedMarkersForCurrentGhostTrack(distToGhost > maxDistToGhost, :);
             detectedMarkersForCurrentGhostTrack = ... 
                 detectedMarkersForCurrentGhostTrack(distToGhost' <= maxDistToGhost, :);
             
            % Correct the estimate of the object's location
                % using the new detection.
            if size(detectedMarkersForCurrentGhostTrack, 1) > 0
                kF = ghostTracks(currentGhostTrackIdx).kalmanFilter;
                %TODO: check to switch motion model
                if kF.motionModel == 0
                    deltas = kF.latest5pos - [kF.latest5pos(end, :); kF.latest5pos(1:end-1, :)];
                    deltas(kF.latestPosIdx+1, :) = [];
                    delta = norm(sum(deltas, 1));
                    dists = pdist2(detectedMarkersForCurrentGhostTrack, kF.x(1:3)');
                    if mean(dists) > 20 && delta > 45 && kF.motionModel == 0 && kF.framesInMotionModel > 15
                        %switch to MotionModel 2
                        kF = switchGhostMM(kF, 2, params);
                        ghostTracks(currentGhostTrackIdx).kalmanFilter = kF;
                    end
                elseif kF.motionModel == 2
                    if norm(kF.x(4:6)) < 2.0 && kF.framesInMotionModel > 1
                        %switch to MotionModel 0
                        kF = switchGhostMM(kF, 0, params);
                        ghostTracks(currentGhostTrackIdx).kalmanFilter = kF;
                    end
                end
                
                
                numDetsGhost = size(detectedMarkersForCurrentGhostTrack, 1);
                % detections are average of assigned observations
                z = mean(detectedMarkersForCurrentGhostTrack, 1)';
                % do kalman correct equations
                if kF.motionModel == 2
                    y = z - kF.H * kF.x;
                    S = kF.H * kF.P * kF.H' + kF.R/numDetsGhost;
                    K = kF.P * kF.H' / S;
                    kF.x = kF.x + K*y;
                    kF.P = (eye(9) - K*kF.H)*kF.P;
                elseif kF.motionModel == 0
                    y = z - kF.H * kF.x;
                    S = kF.H * kF.P * kF.H' + kF.R/numDetsGhost;
                    K = kF.P * kF.H' / S;
                    kF.x = kF.x + K*y;
                    kF.P = (eye(3) - K*kF.H)*kF.P;
                else
                   'updateGhost unexpected motion model!' 
                end

                ghostTracks(currentGhostTrackIdx).kalmanFilter = kF;

                % Update track's age.
                ghostTracks(currentGhostTrackIdx).age = ghostTracks(currentGhostTrackIdx).age + 1;

                % Update visibility.
                ghostTracks(currentGhostTrackIdx).totalVisibleCount = ghostTracks(currentGhostTrackIdx).totalVisibleCount + 1;
                ghostTracks(currentGhostTrackIdx).consecutiveInvisibleCount = 0;
                ghostTracks(currentGhostTrackIdx).trustworthyness = ghostTracks(currentGhostTrackIdx).trustworthyness + numDetsGhost.^2-1;
                
                ghostTracks(currentGhostTrackIdx).kalmanFilter.latest5pos(ghostTracks(currentGhostTrackIdx).kalmanFilter.latestPosIdx+1, :) = ...
                                       ghostTracks(currentGhostTrackIdx).kalmanFilter.x(1:3);
                ghostTracks(currentGhostTrackIdx).kalmanFilter.latestPosIdx = mod(ghostTracks(currentGhostTrackIdx).kalmanFilter.latestPosIdx + 1, 5);
                ghostTracks(currentGhostTrackIdx).kalmanFilter.framesInMotionModel = ghostTracks(currentGhostTrackIdx).kalmanFilter.framesInMotionModel + 1;
            else
                % If detection wasn't assigned to ghost after all
                % increase consecutive invisible count
                ghostTracks(currentGhostTrackIdx).consecutiveInvisibleCount = ghostTracks(currentGhostTrackIdx).consecutiveInvisibleCount + 1;
                ghostTracks(currentGhostTrackIdx).age = ghostTracks(currentGhostTrackIdx).age + 1;
                ghostTracks(currentGhostTrackIdx).trustworthyness = ghostTracks(currentGhostTrackIdx).trustworthyness - 2;
            
                ghostTracks(currentGhostTrackIdx).kalmanFilter.latest5pos(ghostTracks(currentGhostTrackIdx).kalmanFilter.latestPosIdx+1, :) = ...
                                       ghostTracks(currentGhostTrackIdx).kalmanFilter.x(1:3);
                ghostTracks(currentGhostTrackIdx).kalmanFilter.latestPosIdx = mod(ghostTracks(currentGhostTrackIdx).kalmanFilter.latestPosIdx + 1, 5);
                ghostTracks(currentGhostTrackIdx).kalmanFilter.framesInMotionModel = ghostTracks(currentGhostTrackIdx).kalmanFilter.framesInMotionModel + 1;

            end
        end  
        
        % finally delete ghostTracks that were successfully identified
        %ghostTracks(deletedGhostTracks==1) = [];
    end

    function updateAssignedTracksMultiThreaded()
        assignments = double(assignments);
        allAssignedTracksIdx = unique(floor((assignments(:,1)-1)/nMarkers) + 1);
        nAssignedTracks = length(allAssignedTracksIdx);
        
        %Prepare variables to make multi-threading more efficient
        %assignmentIndices = cell(nObjects);
        %detectionIndices = cell(nObjects);
        %detectedMarkersForTracks = cell(nObjects);
        for idxx=1:nObjects
            if ~ismember(idxx, allAssignedTracksIdx)
                continue;
            end
            assignmentIdx = floor((assignments(:,1)-1)/nMarkers) + 1 == idxx;
            detectionIdx = assignments(assignmentIdx,2);
            detectedMarkersForCurrentTrack = detections(detectionIdx, :);
            tracks(idxx).kalmanFilter.z = reshape(detectedMarkersForCurrentTrack, [], 1);
            
            % Update track's age.
            tracks(idxx).age = tracks(idxx).age + 1;
            
            % Update visibility.
            tracks(idxx).totalVisibleCount = tracks(idxx).totalVisibleCount + 1;
            tracks(idxx).consecutiveInvisibleCount = 0;
        end
        updatedKFs = cell(nObjects);
        parfor idxx = 1:nObjects
            if ~ismember(idxx, allAssignedTracksIdx)
                continue;
            end
            %currentTrackIdx = allAssignedTracksIdx(idxx);
            %assignmentsIdx = floor((assignments(:,1)-1)/nMarkers) + 1 == idxx;
            %assignmentsIdx = assignmentIndices{idxx};
            %detectionIdx = assignments(assignmentsIdx,2);
            %detectionIdx = detectionIndices{idxx};
            
            %detectedMarkersForCurrentTrack = detections(detectionIdx, :);
            %%detectedMarkersForCurrentTrack = detectedMarkersForTracks{idxx};
            
            %if size(detectedMarkersForCurrentTrack, 1) > 2 && t > 10
            %   dist = distanceKalman(tracks(currentTrackIdx).kalmanFilter, detectedMarkersForCurrentTrack);
            %   minDist = min(dist, [], 1);
            %   %distToCenter =  sqrt(sum((detectedMarkersForCurrentTrack - tracks(currentTrackIdx).kalmanFilter.x(1:dim)').^2,2));
            %   isValidDetections = minDist < 30;
            %   detectedMarkersForCurrentTrack = detectedMarkersForCurrentTrack(isValidDetections', :);
            %end
            
            % Correct the estimate of the object's location
            % using the new detection.
            %%%%%tracks(idxx).kalmanFilter.z = reshape(detectedMarkersForCurrentTrack, [], 1);
            [updatedKFs{idxx}, ~] = correctKalman(tracks(idxx).kalmanFilter, 1, tracks(idxx).kalmanParams, 0, hyperParams, tracks(idxx).age, params.motionType);
            
            %markerAs = [0,0,0,0];
            %for g=1:size(assignment,2)
            %    markerAs(assignment(1, g)) = 1;
            %end
            
            
            %markerAssignemnts(currentTrackIdx, t, :) = markerAs;
            %falsePositives(t) = falsePositives(t) + (size(detectionIdx,1) - size(assignment,2));
            
            
            
            % Replace predicted bounding box with detected
            % bounding box.
            
            %TODO should be contained in klamanFilter object
            %tracks(trackIdx).center = getCenter(tracks(i).pattern, detectedMarkersForCurrentTrack, tracks(i).kalmanFilter);
            %tracks(trackIdx).markers = detectedMarkersForCurrentTrack;
            
            %             % Update track's age.
            %             tracks(idxx).age = tracks(idxx).age + 1;
            %
            %             % Update visibility.
            %             tracks(idxx).totalVisibleCount = tracks(idxx).totalVisibleCount + 1;
            %             tracks(idxx).consecutiveInvisibleCount = 0;
        end
        for idxx=1:nObjects
            if ~ismember(idxx, allAssignedTracksIdx)
                continue;
            end
            tracks(idxx).kalmanFilter = updatedKFs{idxx};
        end
    end


%% Update Unassigned Tracks
% Mark each unassigned track as invisible, and increase its age by 1.

    function updateUnassignedTracks()
        unassignedTracks = double(unassignedTracks);
        allUnassignedTracksIdx = unique(floor((unassignedTracks-1)/nMarkers) + 1);
        assignedTracks = double(assignedTracks);
        allAssignedTracksIdx = unique(floor((assignedTracks(:,1)-1)/nMarkers) + 1);
        allUnassignedTracksIdx = setdiff(allUnassignedTracksIdx, allAssignedTracksIdx);
        % Remove the assigned tracks from unassigned tracks, when only
        % partially observed
        nUnassignedTracks = length(allUnassignedTracksIdx);
        for i = 1:nUnassignedTracks
            unassignedTrackIdx = allUnassignedTracksIdx(i);
            if tracks(unassignedTrackIdx).age > 0
                tracks(unassignedTrackIdx).age = tracks(unassignedTrackIdx).age + 1;
                tracks(unassignedTrackIdx).kalmanFilter.consecutiveInvisibleCount = tracks(unassignedTrackIdx).kalmanFilter.consecutiveInvisibleCount + 1;
                tracks(unassignedTrackIdx).consecutiveInvisibleCount = tracks(unassignedTrackIdx).kalmanFilter.consecutiveInvisibleCount;
                tracks(unassignedTrackIdx).kalmanFilter.framesInNewMotionModel = tracks(unassignedTrackIdx).kalmanFilter.framesInNewMotionModel + 1;
                tracks(unassignedTrackIdx).kalmanFilter.latest5pos(tracks(unassignedTrackIdx).kalmanFilter.latestPosIdx+1, :) = ...
                    tracks(unassignedTrackIdx).kalmanFilter.mu.X(1:3, 4);
                tracks(unassignedTrackIdx).kalmanFilter.latestPosIdx = mod(tracks(unassignedTrackIdx).kalmanFilter.latestPosIdx + 1, 5);
        
            end
        end
        
        %assignedGhostTracks(:, 1) = assignedGhostTracks(:, 1) - length(tracks)*nMarkers;
        %assignedGhostTracks = double(assignedGhostTracks);
        allAssignedGhostTracksIdx = unique(floor((assignedGhostTracks(:,1)-1)/nMarkers) + 1);
        unassignedGhostTracks(:) = unassignedGhostTracks(:) - length(tracks)*nMarkers;
        unassignedGhostTracks = double(unassignedGhostTracks);
        allUnassignedGhostTracksIdx = unique(floor((unassignedGhostTracks-1)/nMarkers) + 1);
        % remove partially observed tracks from unassigned list
        allUnassignedGhostTracksIdx = setdiff(allUnassignedGhostTracksIdx, allAssignedGhostTracksIdx);
        nUnassignedGhostTracks = length(allUnassignedGhostTracksIdx);
        for i = 1:nUnassignedGhostTracks
            unassignedGhostTrackIdx = allUnassignedGhostTracksIdx(i);
            if ghostTracks(unassignedGhostTrackIdx).age > 0
                ghostTracks(unassignedGhostTrackIdx).age = ghostTracks(unassignedGhostTrackIdx).age + 1;
                ghostTracks(unassignedGhostTrackIdx).consecutiveInvisibleCount = ghostTracks(unassignedGhostTrackIdx).consecutiveInvisibleCount + 1;
                ghostTracks(unassignedGhostTrackIdx).trustworthyness = max(0, ghostTracks(unassignedGhostTrackIdx).trustworthyness - 2);
            
                ghostTracks(unassignedGhostTrackIdx).kalmanFilter.latest5pos(ghostTracks(unassignedGhostTrackIdx).kalmanFilter.latestPosIdx+1, :) = ...
                                       ghostTracks(unassignedGhostTrackIdx).kalmanFilter.x(1:3);
                ghostTracks(unassignedGhostTrackIdx).kalmanFilter.latestPosIdx = mod(ghostTracks(unassignedGhostTrackIdx).kalmanFilter.latestPosIdx + 1, 5);
                ghostTracks(unassignedGhostTrackIdx).kalmanFilter.framesInMotionModel = ghostTracks(unassignedGhostTrackIdx).kalmanFilter.framesInMotionModel + 1;

            end
        end
    end


%% Delete Lost Tracks
% The |deleteLostTracks| function deletes tracks that have been invisible
% for too many consecutive frames. It also deletes recently created tracks
% that have been invisible for too many frames overall.

    function deleteLostTracks(deletedGhostTracks)
        
        invisibleForTooLong = 25;
        invisibleForTooLongGhosts = 10;

        ageThreshold = 10;
        visibilityFraction = 0.5;
        
        ages = [tracks(:).age];
        
        %TODO: do the same for ghost birds
        % delte tracks that drift towards other birds, because their own
        % detections vanished
        nTracks = length(tracks);
        tooCloseBirds = zeros(nTracks,1);
        tooClose = 40;
        positions = NaN*zeros(nTracks,3);
        for i=1:nTracks
           if tracks(i).age > 0
              positions(i,:) = tracks(i).kalmanFilter.mu.X(1:3, 4); 
           end
        end
        distsBirds = triu(squareform(pdist(positions)));
        [rows, cols] = ind2sub(size(distsBirds), find(distsBirds > 0 & distsBirds < tooClose));
        
        for r =1:length(rows)
            bird1Idx = rows(r);
            bird2Idx = cols(r);
            s1 = tracks(bird1Idx).kalmanFilter;
            s2 = tracks(bird2Idx).kalmanFilter;
            if tracks(bird1Idx).kalmanFilter.mu.motionModel == 0 && ...
                    tracks(bird2Idx).kalmanFilter.mu.motionModel == 0
                deltas1 = s1.latest5pos - [s1.latest5pos(end, :); s1.latest5pos(1:end-1, :)];
                deltas1(s1.latestPosIdx+1, :) = [];
                delta1 = sum(norm(deltas1, 1));
                deltas2 = s2.latest5pos - [s2.latest5pos(end, :); s2.latest5pos(1:end-1, :)];
                deltas2(s2.latestPosIdx+1, :) = [];
                delta2 = sum(norm(deltas2, 1));
                
                if delta1 > delta2
                    tooCloseBirds(bird1Idx) = 1;
                else
                    tooCloseBirds(bird2Idx) = 1;
                end
            elseif tracks(bird1Idx).kalmanFilter.mu.motionModel == 2 && ...
                    tracks(bird2Idx).kalmanFilter.mu.motionModel == 0
                tooCloseBirds(bird1Idx) = 1;
            elseif tracks(bird1Idx).kalmanFilter.mu.motionModel == 0 && ...
                    tracks(bird2Idx).kalmanFilter.mu.motionModel == 2
                tooCloseBirds(bird2Idx) = 1;
            elseif tracks(bird1Idx).kalmanFilter.mu.motionModel == 2 && ...
                    tracks(bird2Idx).kalmanFilter.mu.motionModel == 2
                if norm(tracks(bird1Idx).kalmanFilter.mu) > norm(tracks(bird2Idx).kalmanFilter.mu)
                    tooCloseBirds(bird1Idx) = 1;
                else
                    tooCloseBirds(bird2Idx) = 1;
                end
            end
        end
        
        
                % Find the indices of 'lost' tracks.
        lostIdxBool = ( [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong | ... 
                        tooCloseBirds' ) & (ages > 0);
        lostIdx = find(lostIdxBool);
        if ~isempty(lostIdx)
            for i=1:length(lostIdx)
                %mark track as lost/pattern as unassigned
                unassignedPatterns(lostIdx(i)) = 1;
                tracks(lostIdx(i)).age = 0;
                tracks(lostIdx(i)).kalmanFilter.framesInNewMotionModel = 5;
                tracks(lostIdx(i)).kalmanFilter.latest5pos = zeros(5,3);
                tracks(lostIdx(i)).kalmanFilter.latestPosIdx = 0;
        
                estimatedPositions(lostIdx(i), max(1,t-invisibleForTooLong):t-1, :) = NaN;
                estimatedQuats(lostIdx(i), max(1, t-invisibleForTooLong):t-1, :) = NaN;
            end
        end
        
        % Compute the fraction of the track's age for which it was visible.
        ages = [ghostTracks(:).age];
        totalVisibleCounts = [ghostTracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        lostGhostsIdx = [ghostTracks(:).consecutiveInvisibleCount] >= invisibleForTooLongGhosts;% | ( ages < ageThreshold & visibility < visibilityFraction)
        lostGhostsIdx = lostGhostsIdx | deletedGhostTracks';
        ghostTracks(lostGhostsIdx == 1) = [];        
    end


%% Create New Tracks
% Create new tracks from unassigned detections. Assume that any unassigned
% detection is a start of a new track. In practice, you can use other cues
% to eliminate noisy detections, such as size, location, or appearance.

    % see MultipleObjectTracking/createNewTracks()

    
%% Vizualization methods

% This function sets up the figure.
%
    function initializeFigure()
        figure;
        scatter3([minPos(1), maxPos(1)], [minPos(2), maxPos(2)], [minPos(3), maxPos(3)], '*')
        hold on;
        if shouldShowTruth && exist('trueTrajectory', 'var')
            for k = 1:nObjects
                trueTrajectories{k} = plot3(trueTrajectory(k,1,1),trueTrajectory(k,1,2), trueTrajectory(k,1,3), 'Color', colorsTrue(k,:));
            end
        end
        
        for k = 1:nObjects
            birdsTrajectories{k} = plot3(NaN, NaN, NaN, 'Color', colorsPredicted(k,:));
        end
        dets = squeeze(D(1,:,:));
        markersForVisualization{1} = plot3(dets(:,1),dets(:,2), dets(:,3), '*', 'MarkerSize', 5, 'MarkerEdgeColor', [0.5; 0.5; 0.5]);
        ghostBirdsVis{1} = plot3(NaN*zeros(12,1), NaN*zeros(12,1), NaN*zeros(12,1), 'o', 'MarkerSize', 13, 'MarkerEdgeColor', [0.5; 0.5; 0.5]);
        for k = 1:nObjects
            %birdsPositions{k} = plot3(NaN, NaN, NaN, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', colors(k,:));
            for n = 1:nMarkers
                markerPositions{k,n} = plot3(NaN, NaN, NaN, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', colorsPredicted(k,:));
                viconMarkerPositions{k,n} = plot3(NaN, NaN, NaN, 'square', 'MarkerSize', 12, 'MarkerEdgeColor', colorsTrue(k,:));
            end
        end
        
        
        grid on;
        %axis equal;
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
                    
                    if strcmp(model, 'LieGroup')
                        xPos = tracks(k).kalmanFilter.mu.X(1,4);
                        yPos = tracks(k).kalmanFilter.mu.X(2,4);
                        zPos = tracks(k).kalmanFilter.mu.X(3,4);
                    else
                        xPos = tracks(k).kalmanFilter.x(1);
                        yPos = tracks(k).kalmanFilter.x(2);
                        zPos = tracks(k).kalmanFilter.x(3);
                    end
                    
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
                    birdsTrajectories{k}.Color = colorsPredicted(k,:);
                    
                    pattern = tracks(k).kalmanFilter.pattern;
                    if strcmp(model, 'LieGroup')
                        rotMat = tracks(k).kalmanFilter.mu.X(1:3, 1:3);
                    else
                        if strcmp(params.motionType, 'constAcc')
                            quat = tracks(k).kalmanFilter.x(10:13);
                        else
                            quat = tracks(k).kalmanFilter.x(7:10);
                        end
                        rotMat = Rot(quat);
                        
                    end
                    rotatedPattern = (rotMat * pattern')';
                    
                    for n = 1:nMarkers
                        markerPositions{k,n}.XData = xPos + rotatedPattern(n,1);
                        markerPositions{k,n}.YData = yPos + rotatedPattern(n,2);
                        markerPositions{k,n}.ZData = zPos + rotatedPattern(n,3);
                        if tracks(k).kalmanFilter.flying > 0
                            markerPositions{k,n}.Marker = 's';
                        else
                            markerPositions{k,n}.Marker = 'o';
                        end
                    end
                else
                   for n=1:nMarkers
                      markerPositions{k,n}.XData = NaN;
                      markerPositions{k,n}.YData = NaN; 
                      markerPositions{k,n}.ZData = NaN; 
                   end
                   birdsTrajectories{k}.Color = colorsTrue(k,:);
                end
            end
        end
        
        dets = squeeze(D(t, :, :));
        markersForVisualization{1}.XData = dets(:,1);
        markersForVisualization{1}.YData = dets(:,2);
        markersForVisualization{1}.ZData = dets(:,3);
        
        ghostsPos = NaN*zeros(12,3);
        for n=1:length(ghostTracks)
            ghostsPos(n, :) = ghostTracks(n).kalmanFilter.x(1:3)';
        end
        ghostBirdsVis{1}.XData = ghostsPos(:, 1);
        ghostBirdsVis{1}.YData = ghostsPos(:, 2);
        ghostBirdsVis{1}.ZData = ghostsPos(:, 3);

        drawnow
        %pause(0.1)
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
