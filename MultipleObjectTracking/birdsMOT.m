%% load data
load('rawDetections.mat')
load('D_labeled.mat')

%% prepare data
offset = 130;
% TODO why 'offset - 1'? ???
formattedData = formattedData(1011+offset-1:end, :,:);
initialData = D_labeled(offset,:);
viconData = D_labeled(offset-1:end,:);
viconTrajectories = zeros(size(viconData,1), 10, 3);
viconTrajectories(:,:,1) = viconData(:,5:7:end);
viconTrajectories(:,:,2) = viconData(:,6:7:end);
viconTrajectories(:,:,3) = viconData(:,7:7:end);
viconTrajectories = permute(viconTrajectories, [2,1,3]);

viconOrientation = zeros(size(viconData,1), 10, 4);
viconOrientation(:,:,2) = viconData(:,1:7:end);
viconOrientation(:,:,3) = viconData(:,2:7:end);
viconOrientation(:,:,4) = viconData(:,3:7:end);
viconOrientation(:,:,1) = viconData(:,4:7:end);
viconOrientation = permute(viconOrientation, [2,1,3]);

% TODO is mirroring taken car of?
initialStates = zeros( length(initialData)/7, 3+3+4+4 );
for i = 1:size(initialStates,1)
   initialStates(i,1:3) = initialData( 7*(i-1)+5:i*7 );
   initialStates(i,4:6) = zeros(1,3);
   initialStates(i,7:10) = initialData(7*(i-1)+1:(i-1)*7+4);
end

initialStates = initialStates( sum(isnan(initialStates),2) == 0, :);

%% Get patterns
allPatterns = read_patterns('tracking/datasets/framework');
patterns = zeros(10,4,3);
patternNames = {};
for i=1:length(allPatterns)
    patterns(i,:,:) = allPatterns(i).pattern;
    patternNames{i} = allPatterns(i).name;
end
    
    
%% test MOT
stdHyperParams.doFPFiltering = 1;
stdHyperParams.adaptiveNoise = 1;
stdHyperParams.lambda = 0;
stdHyperParams.simplePatternMatching = 0;

stdHyperParams.costOfNonAsDtTA = 70;
stdHyperParams.certaintyFactor = 1;
stdHyperParams.useAssignmentLength = 1;
stdHyperParams.whenFPFilter = 70;
stdHyperParams.thresholdFPFilter = 50;
stdHyperParams.costOfNonAsDtMA = 10;
stdHyperParams.eucDistWeight = 1/3;
stdHyperParams.posNoise = 30;
stdHyperParams.motNoise = 30;
stdHyperParams.quatNoise = 0.5;
stdHyperParams.quatMotionNoise = 0.05;
stdHyperParams.measurementNoise = 100;
stdHyperParams.certaintyScale = 3;
quatMotionType = 'brownian';
[estimatedPositions, estimatedQuats] = ownMOT(formattedData, patterns, patternNames ,1 , initialStates, 10, 0, viconTrajectories, viconOrientation, quatMotionType, stdHyperParams);
%% Evaluate tracking performance 
% Plot the estimation error of the positions and orientations

%performanceVisualization(estimatedPositions, positions, estimatedQuats, quats);

%Get VICON's estimates
D_offset = D_labeled(offset-1:end,:);
VICONX = D_offset(:,5:7:end);
VICONY = D_offset(:,6:7:end);
VICONZ = D_offset(:,7:7:end);
VICONq2 = D_offset(:,1:7:end);
VICONq3 = D_offset(:,2:7:end);
VICONq4 = D_offset(:,3:7:end);
VICONq1 = D_offset(:,4:7:end);

VICONPos = cat(3, VICONX, VICONY, VICONZ);
VICONPos = permute(VICONPos, [2 1 3]);

VICONquat = cat(3, VICONq1, VICONq2, VICONq3, VICONq4);
VICONquat = permute(VICONquat, [2 1 3]);
if size(VICONPos,2) < size(estimatedPositions,2)
    estimatedPositions = estimatedPositions(:,1:size(VICONPos,2),:);
    estimatedQuats = estimatedQuats(:, 1:size(VICONquat,2), :);
else
    VICONPos = VICONPos(:,1:size(estimatedPositions,2),:);
    VICONquat = VICONquat(:,1:size(estimatedPositions,2),:);
end
%%
performanceVisualization(estimatedPositions, VICONPos, estimatedQuats, VICONquat, patterns, 1);

%% save results
%save('trackingResultsLowNoise', 'estimatedPositions', 'estimatedQuats', 'VICONPos', 'VICONquat', 'patterns')

%% load results and display
load('trackingResults')

%% compare old vs new with FP filter
hold on
k=3
%for k=1:10
plot(estPos(k,:,3), 'b')
plot(estimatedPositions(k,:,3), 'r')
%end
hold off
%%
D_offset = D_labeled(offset:end,:);
%D_offset = table2array(correctedVICON);
VICONX = D_offset(:,5:7:end);
VICONY = D_offset(:,6:7:end);
VICONZ = D_offset(:,7:7:end);
VICONq2 = D_offset(:,1:7:end);
VICONq3 = D_offset(:,2:7:end);
VICONq4 = D_offset(:,3:7:end);
VICONq1 = D_offset(:,4:7:end);

VICONPos = cat(3, VICONX, VICONY, VICONZ);
VICONPos = permute(VICONPos, [2 1 3]);

VICONquat = cat(3, VICONq1, VICONq2, VICONq3, VICONq4);
VICONquat = permute(VICONquat, [2 1 3]);

%%
t0=1700;

%vizRes(formattedData(t0:end,:,:), patterns, estimatedPositions(:,t0:end,:), estimatedQuats(:,t0:end,:), 1, VICONPos(:, t0:end, :), VICONquat(:, t0:end, :))
vizRes(formattedData(t0:end,:,:), patterns, estPos(:,t0:end,:), estQuats(:,t0:end,:), 1, estimatedPositions(:, t0:end, :), estimatedQuats(:, t0:end, :))
