%%
if batchStartupOptionUsed
    cd ../..
end
path = pwd;
addpath(genpath([path, '/', 'multiple_object_tracking_project']))
%%
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

%files = {'Starling_Trials_10-12-2019_08-15-00.txt', ...
%         'Starling_Trials_10-12-2019_08-30-00.txt', ...
%         'Starling_Trials_10-12-2019_08-45-00.txt'};
dirName = 'multiple_object_tracking_project/datasets';
processedFileName = 'multiple_object_tracking_project/datasets/processedFiles.txt';
files = getUnprocessedFiles(dirName, processedFileName);
patternDirectoryName = 'multiple_object_tracking_project/datasets/session8';


disp('starting to process files in parallel!')
                     
parpool(8)
parfor i=1:length(files)
    tic
    fprintf(['Processing File: ', num2str(i), '\n'])
    try
        birdsMOT(files{i}, patternDirectoryName, stdHyperParams);
        fid = fopen('multiple_object_tracking_project/datasets/processedFiles.txt', 'a');
        fs = strsplit(files{i}, '/');
        f = fs{end};
        fprintf(fid, '%s\n', f);
        fclose(fid);
    catch MExc
        disp(MExc.message)
    end
    toc
end

delete(gcp('nocreate'))


disp('Done processing all files')
% %%
% 
% load(['datasets/Starling_Trials_10-12-2019_08-45-00.mat']);
% patternsPlusNames = read_patterns(patternDirectoryNames{1});
% patterns = zeros(length(patternsPlusNames),4,3);
% for i=1:length(patternsPlusNames)
%     patterns(i,:,:) = patternsPlusNames(i).pattern;
% end
% 
% vizParams.vizSpeed = 10;
% vizParams.keepOldTrajectory = 0;
% vizParams.vizHistoryLength = 500;
% vizParams.startFrame = 1;
% vizParams.endFrame = length(estPosRes{3});
% vizRes(formattedData(1:length(estPosRes{3}), :, :), patterns, estPosRes{3}, estQuatRes{3}, vizParams, 0)
