%%
%ver
%if batchStartupOptionUsed
%    cd ../..
%end
%cd ../..
%path = pwd;
%addpath(genpath([path, '/', 'multiple_object_tracking_project']))



%% load data and patterns

%flock = 4;
%recordingName = 'Starling_Trials_11-01-2020_14-00-00';
flock = 3;
recordingName = 'Starling_Trials_15-12-2019_08-30-00';%'Starling_Trials_14-01-2020_14-00-00'; %
dataFolder = 'multiple_object_tracking_project/datasets';
patternDirectoryName = [dataFolder, '/flock', num2str(flock), '/patterns'];

%% test MOT
stdHyperParams.doFPFiltering = 1;
stdHyperParams.adaptiveNoise = 1;
stdHyperParams.lambda = 0;
stdHyperParams.simplePatternMatching = 0;

stdHyperParams.costOfNonAsDtTA = 50;%85 %session8: 50, alt: flying birds get lower assignment cost
stdHyperParams.certaintyFactor = 1;
stdHyperParams.useAssignmentLength = 1;
stdHyperParams.minAssignmentThreshold = 30; %35%30;
stdHyperParams.ghostFPFilterDist = 65;
stdHyperParams.costOfNonAsDtMA = 10;
stdHyperParams.eucDistWeight = 1/10;%1/3;

% params regarding: process noise, i.e. how reliable the predictions of the
% motion models are
% params for constant acceleration model:
stdHyperParams.posNoise = 30;%%110;%60;%50
stdHyperParams.motNoise = 10;%;1;%5;%10
stdHyperParams.accNoise = 1;%0.1;%1;%3
% params for brownian motion model
stdHyperParams.posNoiseBrownian = 80; 
% params shared across all motion models
stdHyperParams.quatNoise = 0.2;
stdHyperParams.quatMotionNoise = 1; % not used, brownian motion model for rotation works best

% params regarding: measurement noise i.e. determine how reliable
% measurements are
stdHyperParams.measurementNoise = 50;%50
stdHyperParams.certaintyScale = 5;%6.5

% params regarding: initialization of ghost birds
%minimal distance for new ghost birds to other (ghost) birds that has to be free.
stdHyperParams.minDistToBird = [95 65 50 40]; % number at index i is used when i detections are present
% params regarding: initialization of birds from ghost bird when all others are known
stdHyperParams.minTrustworthyness = 10;

% params regarding: initialization of birds
stdHyperParams.initThreshold = 0.5;%0.85;
stdHyperParams.initThreshold4 = 2.75;
stdHyperParams.costDiff = 1.5;
stdHyperParams.costDiff4 = 1.25;

stdHyperParams.initThresholdTight = 0.2;%0.65
stdHyperParams.initThreshold4Tight = 1.35;
stdHyperParams.costDiffTight = 2;
stdHyperParams.costDiff4Tight = 1.25;

stdHyperParams.patternSimilarityThreshold = 1.25;%1;

stdHyperParams.modelType = 'LieGroup';
quatMotionType = 'brownian';



fprintf('Starting to track!\n')

%profile on;
beginningFrame = 1;
endFrame = -1;
stdHyperParams.visualizeTracking = 1;

[estPos, estQuat, dets, patterns, patternNames, ~, ~] = birdsMOT([dataFolder, '/flock', num2str(flock), '/', recordingName], ...
                                    [dataFolder, '/flock', num2str(flock)], stdHyperParams, flock, ...
                                    beginningFrame, endFrame);



%%
vizParams.vizSpeed = 10;
vizParams.keepOldTrajectory = 0;
vizParams.vizHistoryLength = 500;
vizParams.startFrame = 1;
vizParams.endFrame = endFrame;
%revIdx = sort(1:length(formattedData), 'descend');
%dets = formattedData(revIdx, :, :);
%reverseIdx = sort(1:size(estimatedPositionsRev, 2), 'descend');
%vizRes(dets, patterns, estimatedPositionsBackward, estimatedQuatsBackward, vizParams, 0)
%%
vizRes(dets, patterns, estPos, estQuat, vizParams, 0)

%%
%
%vizRes(dets, patterns, estimatedPositions, estimatedQuats, vizParams, 0)

%vizParams.vizSpeed = 5;
%vizRes(formattedData, patterns, viconTrajectories, viconOrientation, vizParams, 0)


%%
%augPos = postProcessing(estPos, ghostTracks, patterns);
%vizRes(dets, patterns, augPos, estQuat, vizParams, 0)

exportToCSV('testExportSECOND.csv', estPos, estQuat, patternNames, 1)
