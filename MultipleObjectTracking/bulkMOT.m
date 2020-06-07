stdHyperParams.doFPFiltering = 1;
stdHyperParams.adaptiveNoise = 1;
stdHyperParams.lambda = 0;
stdHyperParams.simplePatternMatching = 0;

stdHyperParams.costOfNonAsDtTA = 50;
stdHyperParams.certaintyFactor = 1;
stdHyperParams.useAssignmentLength = 1;
stdHyperParams.minAssignmentThreshold = 30; %35%30;
stdHyperParams.ghostFPFilterDist = 65;
stdHyperParams.costOfNonAsDtMA = 10;
stdHyperParams.eucDistWeight = 1/4;%1/3;

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
stdHyperParams.minDistToBird = [95 80 50 40]; % number at index i is used when i detections are present
% params regarding: initialization of birds from ghost bird when all others are known
stdHyperParams.minTrustworthyness = 10;

% params regarding: initialization of birds
stdHyperParams.initThreshold = 0.75;%0.85;
stdHyperParams.initThreshold4 = 2.5;
stdHyperParams.patternSimilarityThreshold = 1.25;%1;

stdHyperParams.modelType = 'LieGroup';

stdHyperParams.visualizeTracking = 0;

%%
%TODO get list from folder

files = {'Starling_Trials_10-12-2019_08-15-00.txt', ...
         'Starling_Trials_10-12-2019_08-30-00.txt', ...
         'Starling_Trials_10-12-2019_08-45-00.txt'};
patternDirectoryNames = {'datasets/session8', 'datasets/session8', 'datasets/session8'};

estPosRes = {};
estQuatsRes = {};
certRes = {};
ghostTracksRes = {};

parpool(3)
parfor i=1:length(files)
    [estPosRes{i}, estQuatRes{i}, certRes{i}, ghostTracksRes{i}] = ...
            birdsMOT(files{i}, patternDirectoryNames{i}, stdHyperParams);
end

delete(gcp('nocreate'))