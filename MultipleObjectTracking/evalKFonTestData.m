% This script evaluates the Kalman Filter approach on the test data on
% which the neural network was evaluated.

%% load data
%load('tracking/modernMethods/data/artificial_none.mat')
load('tracking/modernMethods/data/birdlike_higher.mat')

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
% stdHyperParams.doFPFiltering = 1;
% stdHyperParams.adaptiveNoise = 1;
% stdHyperParams.lambda = 0;
% stdHyperParams.simplePatternMatching = 0;
% 
% stdHyperParams.costOfNonAsDtTA = 0.7;
% stdHyperParams.certaintyFactor = 50;
% stdHyperParams.useAssignmentLength = 1;
% stdHyperParams.minAssignmentThreshold = 0.9;
% stdHyperParams.thresholdFPFilter = 0.7;
% stdHyperParams.costOfNonAsDtMA = 0.095;
% stdHyperParams.eucDistWeight = 1/8;
% stdHyperParams.posNoise = 0.002;
% stdHyperParams.motNoise = 0.002;
% stdHyperParams.accNoise = 0.001;
% stdHyperParams.quatNoise = 0.5;
% %stdHyperParams.quatMotionNoise = 0.5;
% stdHyperParams.measurementNoise = 2;
% stdHyperParams.certaintyScale = 0.3;
% stdHyperParams.visualizeTracking = 1;
% quatMotionType = 'brownian';
stdHyperParams.doFPFiltering = 1;
stdHyperParams.adaptiveNoise = 1;
stdHyperParams.lambda = 0;
stdHyperParams.simplePatternMatching = 0;

stdHyperParams.costOfNonAsDtTA = 0.7;
stdHyperParams.certaintyFactor = 50;
stdHyperParams.useAssignmentLength = 1;
stdHyperParams.minAssignmentThreshold = 0.7;
stdHyperParams.costOfNonAsDtMA = 0.095;
stdHyperParams.eucDistWeight = 1/10;
stdHyperParams.posNoise = 0.01;
stdHyperParams.motNoise = 0.01;
stdHyperParams.accNoise = 0.01;
stdHyperParams.quatNoise = 0.02;
stdHyperParams.quatMotionNoise = -1;
stdHyperParams.measurementNoise = 0.2;
stdHyperParams.certaintyScale = 0.3;
stdHyperParams.visualizeTracking = 0;
quatMotionType = 'brownian';
stdHyperParams.modelType = 'extended';

%%
exN = 9;
if length(size(patterns)) == 4
    pats = reshape(patterns(1, exN, :, :), 1, 4, 3);
else
    pats = reshape(patterns(exN, :, :), 1, 4, 3);
end

size(pats)
p = squeeze(pos(:, exN, :));
%q = squeeze(quats(:, 1, :));

qMat = squeeze(quats(:, exN, :, :));
q = rotm2quat(squeeze(qMat(1, :, :))); 

%initialStates = [p(1, :) 0 0 0 rotm2quat(reshape(q(1, :), 3, 3)') 0 0 0 0];
initialStates.pos = p(1, :);
initialStates.velocity = [0 0 0];
initialStates.acceleration = [0 0 0];
initialStates.quat = q(1, :);

size(initialStates)
[estimatedPositions, estimatedQuats] = ownMOT(squeeze(D(1:1:100, exN, :, :)), pats, {'pat'} ,1 , initialStates, 1, 0, -1, -1, quatMotionType, stdHyperParams);
save('tracking/modernMethods/KF_higher.mat', 'estimatedPositions', 'estimatedQuats')
estimatedQuats = squeeze(estimatedQuats);
for t=1:100
   estimatedQuats(t, :) = estimatedQuats(t, :) / sqrt(sum((estimatedQuats(t, :).^2)));
end

figure;
hold on;
plot(squeeze(estimatedPositions))
plot(p(1:1:end, :))
hold off;
%vizRes(squeeze(D(:, exN, :,:)), pats, estimatedPositions, estimatedQuats, 1, permute(pos(:, exN, :), [2, 1, 3]), permute(quats(:, exN, :), [2, 1, 3]))
%%
N = 1000;
predPos = zeros(N, T, 3);
predQuats = zeros(N, T, 4);
errorPlot = zeros(N, T);
for n=1:N
    n
    if length(size(patterns)) == 4
        pats = reshape(patterns(1, n, :, :), 1, 4, 3);
    else
        pats = reshape(patterns(n, :, :), 1, 4, 3);
    end
    p = squeeze(pos(:, n, :));
    q = squeeze(quats(:, n, :, :));
    initialStates.pos = p(1,:);
    initialStates.velocity = [0 0 0];
    initialStates.acceleration = [0 0 0];
    initialStates.quat = rotm2quat(squeeze(q(1, :, :)));
    %initialStates = [p(1, :) 0 0 0 rotm2quat(reshape(q(1, :), 3, 3)') 0 0 0 0];
    [estimatedPositions, estimatedQuats] = ownMOT(squeeze(D(:, n, :, :)), pats, {'pat'} ,1 , initialStates, 1, 0, -1, -1, quatMotionType, stdHyperParams);
    for t=1:T
       pat = squeeze(patterns(n, :, :)); 
       eQ = squeeze(estimatedQuats(1, t, :));
       eQ = eQ / norm(eQ);
       eRM = quat2rotm(eQ');
       errorPlot(n, t) = mean(( (eRM * pat')' - (squeeze(quats(t, n, :, :)) * pat')' ).^2, 'all');
    end
    predPos(n, :, :) = squeeze(estimatedPositions);
    predQuats(n, :, :) = squeeze(estimatedQuats);
end

%%

CI95 = zeros(T, 1);

for t=1:T
   data = sort(errorPlot(:, t));
   idx = floor(N*0.95);
   CI95(t) = data(idx);
end

CI90 = zeros(T, 1);

for t=1:T
   data = sort(errorPlot(:, t));
   idx = floor(N*0.9);
   CI90(t) = data(idx);
end

CI99 = zeros(T, 1);

for t=1:T
   data = sort(errorPlot(:, t));
   idx = floor(N*0.99);
   CI99(t) = data(idx);
end

figure;
plot(smoothdata(errorPlot', 'gauss', 10), 'color', [0, 0, 1, 0.15], 'LineWidth', 0.25)
hold on;
plot(CI90, 'LineWidth', 3, 'Color', [0.9290    0.6940    0.1250], 'LineStyle', '--')
plot(CI95, 'LineWidth', 3, 'Color', [0.7500         0    0.7500], 'LineStyle', ':')
plot(CI99, 'LineWidth', 3, 'Color', [0.8500    0.3250    0.0980], 'LineStyle', '-.')



%plot(smoothdata(smoothdata(errorPlot', 'movmedian', 10), 'movmean', 10))
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
save('test_EKF_res', 'errorPlot')

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
