% This script evaluates the Kalman Filter approach on the test data on
% which the neural network was evaluated.

%% load data
load('tracking/modernMethods/data/artificial_none.mat')
load('tracking/modernMethods/data/artificial_higher.mat')

detections = detections;% / 4.5565;
patterns = patterns;% / 4.5565;
pos = pos;% / 4.5565;
numDets = 6;

%% prepare data
[T, N, ~] = size(pos);
dets = zeros(T, N, numDets*3) * NaN;

detections = permute(reshape(detections, [T, N, 4, numDets]), [1, 2, 4, 3]);
squeeze(detections(1, 1, :, :))
lostIdx = detections(:, :, :, 4) == 1;
size(lostIdx)
squeeze(lostIdx(1, 1, :))
lostIdx = repmat(lostIdx, [1, 1, 1, 3]);
size(lostIdx)
D = detections(:, :, :, 1:3);
D(lostIdx) = NaN;
size(D)

%% Get patterns
%allPatterns = read_patterns('tracking/datasets/framework');
%patterns = zeros(10,4,3);
%patternNames = {};
%for i=1:length(allPatterns)
%    patterns(i,:,:) = allPatterns(i).pattern;
%    patternNames{i} = allPatterns(i).name;
%end
    
    
%% test MOT
stdHyperParams.doFPFiltering = 1;
stdHyperParams.adaptiveNoise = 1;
stdHyperParams.lambda = 0;
stdHyperParams.simplePatternMatching = 0;

stdHyperParams.costOfNonAsDtTA = 0.8;
stdHyperParams.certaintyFactor = 50;
stdHyperParams.useAssignmentLength = 1;
stdHyperParams.whenFPFilter = 0.6;
stdHyperParams.thresholdFPFilter = 0.7;
stdHyperParams.costOfNonAsDtMA = 0.095;
stdHyperParams.eucDistWeight = 1/8;
stdHyperParams.posNoise = 0.25;
stdHyperParams.motNoise = 0.25;
stdHyperParams.quatNoise = 0.075;
stdHyperParams.quatMotionNoise = 0.05;
stdHyperParams.measurementNoise = 2;
stdHyperParams.certaintyScale = 0.3;
quatMotionType = 'brownian'

%%
exN = 4;
if length(size(patterns)) == 4
    pats = reshape(patterns(1, exN, :, :), 1, 4, 3);
else
    pats = reshape(patterns(exN, :, :), 1, 4, 3);
end

size(pats)
p = squeeze(pos(:, exN, :));
%q = squeeze(quats(:, 1, :));

qMat = squeeze(quats(:, exN, :));
qMat = permute(reshape(qMat, [100, 3, 3]), [3, 2, 1]);
q = rotm2quat(qMat); 

%initialStates = [p(1, :) 0 0 0 rotm2quat(reshape(q(1, :), 3, 3)') 0 0 0 0];
initialStates = [p(1, :) 0 0 0 q(1, :) 0 0 0 0];

size(initialStates)
[estimatedPositions, estimatedQuats] = ownMOT(squeeze(D(:, exN, :, :)), pats, {'pat'} ,1 , initialStates, 1, 0, -1, -1, quatMotionType, stdHyperParams);
save('tracking/modernMethods/KF_higher.mat', 'estimatedPositions', 'estimatedQuats')
figure;
hold on;
plot(squeeze(estimatedPositions))
plot(p)
hold off;
%vizRes(squeeze(D(:, exN, :,:)), pats, estimatedPositions, estimatedQuats, 1, permute(pos(:, exN, :), [2, 1, 3]), permute(quats(:, exN, :), [2, 1, 3]))
%%
predPos = zeros(250, T, 3);
predQuats = zeros(250, T, 4);
profile on
for n=1:100
    n
    
    if length(size(patterns)) == 4
        pats = reshape(patterns(1, n, :, :), 1, 4, 3);
    else
        pats = reshape(patterns(n, :, :), 1, 4, 3);
    end
    p = squeeze(pos(:, n, :));
    q = squeeze(quats(:, n, :));
    initialStates = [p(1, :) 0 0 0 rotm2quat(reshape(q(1, :), 3, 3)') 0 0 0 0];
    [estimatedPositions, estimatedQuats] = ownMOT(squeeze(D(:, n, :, :)), pats, {'pat'} ,1 , initialStates, 1, 0, -1, -1, quatMotionType, stdHyperParams);
    predPos(n, :, :) = squeeze(estimatedPositions);
    predQuats(n, :, :) = squeeze(estimatedQuats);
end
profile viewer
% %%
% predPos = zeros(floor(N/23), T, 3);
% predQuats = zeros(floor(N/23), T, 4);
% for n=23:23:N
%     pats = reshape(patterns(1, n, :, :), 1, 4, 3);
%     p = squeeze(pos(:, n, :));
%     q = squeeze(quats(:, n, :));
%     initialStates = [p(1, :) 0 0 0 rotm2quat(reshape(q(1, :), 3, 3)') 0 0 0 0];
%     [estimatedPositions, estimatedQuats] = ownMOT(squeeze(D(:, n, :, :)), pats, {'pat'} ,1 , initialStates, 1, 0, -1, -1, quatMotionType, stdHyperParams);
%     predPos(floor(n/23), :, :) = squeeze(estimatedPositions);
%     predQuats(floor(n/23), :, :) = squeeze(estimatedQuats);
% end
%%
save('test_KFresult_birdlike_none.mat', 'predPos', 'predQuats')

%%
load('test_KFresult_artificial_higher.mat')
figure;
plot(squeeze(predPos(1, :, :)))






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
