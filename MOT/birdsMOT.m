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
idx = 1;
for i=1:length(allPatterns)
    patterns(idx,:,:) = allPatterns(i).pattern;
    idx = idx + 1;
end
    
    
%% test MOT
quatMotionType = 'brownian';
[estimatedPositions, estimatedQuats] = ownMOT(formattedData(1:3000,:,:), patterns, initialStates, 10, viconTrajectories, viconOrientation, quatMotionType);

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

performanceVisualization(estimatedPositions, VICONPos, estimatedQuats, VICONquat, patterns);

%% save results
%save('trackingResults', 'estimatedPositions', 'estimatedQuats', 'VICONPos', 'VICONquat', 'patterns')

%% load results and display
load('trackingResults')
vizRes(formattedData(1:3000,:,:), patterns, estimatedPositions, estimatedQuats, 1, VICONPos, VICONquat)
