function vizRes(D, patterns, estimatedPositions, estimatedQuats, vizParams, shouldShowTruth, trueTrajectory, trueOrientation)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

keepOldTrajectory = vizParams.keepOldTrajectory;
vizHistoryLength = vizParams.vizHistoryLength;
vizSpeed = vizParams.vizSpeed;
startFrame = vizParams.startFrame;
endFrame = vizParams.endFrame;

nObjects = size(patterns, 1);
nMarkers = size(patterns, 2);

% skip some frames if speed > 1
if vizSpeed > 1
    vizSpeed = ceil(vizSpeed);
    estimatedQuats = estimatedQuats(:, startFrame:vizSpeed:endFrame, :);
    estimatedPositions = estimatedPositions(:, startFrame:vizSpeed:endFrame, :);
    D = D(startFrame:vizSpeed:endFrame, :, :);
else
    estimatedQuats = estimatedQuats(:, startFrame:endFrame, :);
    estimatedPositions = estimatedPositions(:, startFrame:endFrame, :);
    D = D(startFrame:endFrame, :, :);
end

detsForVisualization = cell(nObjects,1);
birdsTrajectories = cell(nObjects,1);
trueTrajectories = cell(nObjects,1);
%birdsPositions = cell(nObjects,1);
markerPositions = cell(nObjects, nMarkers);
viconMarkerPositions = cell(nObjects, nMarkers);

colorsPredicted = distinguishable_colors(nObjects);
colorsTrue = (colorsPredicted + 2) ./ (max(colorsPredicted,[],2) +2);


maxX = max(estimatedPositions(:, :, 1), [], 'all');
maxY = max(estimatedPositions(:, :, 2), [], 'all');
maxZ = max(estimatedPositions(:, :, 3), [], 'all');
minX = min(estimatedPositions(:, :, 1), [], 'all');
minY = min(estimatedPositions(:, :, 2), [], 'all');
minZ = min(estimatedPositions(:, :, 3), [], 'all');


figure;
scatter3([maxX, minX], [maxY, minY], [maxZ, minZ]);
hold on;
%axis equal;
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
    detsForVisualization{k} = plot3(dets(:,1),dets(:,2), dets(:,3), '*', 'MarkerSize', 5, 'MarkerEdgeColor', [0.5; 0.5; 0.5]);
    %birdsPositions{k} = plot3(NaN, NaN, NaN, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', colors(k,:));
    for n = 1:nMarkers
        markerPositions{k,n} = plot3(NaN, NaN, NaN, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', colorsPredicted(k,:));
        viconMarkerPositions{k,n} = plot3(NaN, NaN, NaN, 'square', 'MarkerSize', 12, 'MarkerEdgeColor', colorsTrue(k,:));
    end
end


grid on;
%axis equal;
axis manual;

%TODO loop t over all timesteps
for t=1:min(size(D,1), size(estimatedPositions,2))
    t
    for k = 1:nObjects
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
           
        
            pattern = squeeze(patterns(k,:,:));
            trueRotMat = quat2rotm(squeeze(trueOrientation(k, t, :)));
            %trueRotMat = Rot(squeeze(trueOrientation(k, t, :)));
            trueRotatedPattern = (trueRotMat * pattern')';
            
            for n = 1:nMarkers
                viconMarkerPositions{k,n}.XData = trueTrajectory(k, t, 1) + trueRotatedPattern(n,1);
                viconMarkerPositions{k,n}.YData = trueTrajectory(k, t, 2) + trueRotatedPattern(n,2);
                viconMarkerPositions{k,n}.ZData = trueTrajectory(k, t, 3) + trueRotatedPattern(n,3);
            end
        end
        
        xPos = estimatedPositions(k,t,1);
        yPos = estimatedPositions(k,t,2);
        zPos = estimatedPositions(k,t,3);
        
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
        
        pattern = squeeze(patterns(k,:,:));
        quat = squeeze(estimatedQuats(k,t,:));
        rotMat = quat2rotm(quat');
        %rotMat = Rot(quat);
        rotatedPattern = (rotMat * pattern')';
        
        for n = 1:nMarkers
            
            markerPositions{k,n}.XData = xPos + rotatedPattern(n,1);
            markerPositions{k,n}.YData = yPos + rotatedPattern(n,2);
            markerPositions{k,n}.ZData = zPos + rotatedPattern(n,3);
            
        end
        dets = squeeze(D(t,(k-1)*nMarkers+1:k*nMarkers,:));
        detsForVisualization{k}.XData = dets(:,1);
        detsForVisualization{k}.YData = dets(:,2);
        detsForVisualization{k}.ZData = dets(:,3);
    end
    drawnow
    if vizSpeed < 1
        pause(1-vizSpeed);
    end
end
end

