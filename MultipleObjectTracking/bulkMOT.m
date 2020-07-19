%% add relevant functions to path, when working on server

% this MATLAB built-in requires at least version R2019a 
%if batchStartupOptionUsed
    cd ../..
%end
path = pwd;
addpath(genpath([path, '/', 'multiple_object_tracking_project']))

%%
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

stdHyperParams.visualizeTracking = 0;


%% Read files that need to processed
% Files to be processed must be in this directory
dirName = 'multiple_object_tracking_project/datasets';
flock = 2;
flockName = ['flock', num2str(flock)];
% This file should contain the names of all .txt files that already have been processed 
processedFileName = [dirName, '/', flockName , '/processedFiles.txt'];
files = getUnprocessedFiles([dirName, '/', flockName], processedFileName);
% The .vsk files specifiying the patterns for the recordings need to be
% stored in this directory
patternDirectoryName = [dirName, '/', flockName, '/patterns'];

%% Process files in parallel 
disp('starting to process files in parallel!')

% number of CPUs to be used
parpool(32);
parfor i=1:length(files)
    tic
    fprintf(['Processing File ', num2str(i), ': ', files{i}, '\n'])
    try
        % 'birdsMOT' does all the work, i.e. read data from .txt to .mat
        % and then runs the MOT algorithm and stores the results in the
        % 'RESULTS' folder
        birdsMOT(files{i}, [dirName, '/', flockName], stdHyperParams, flock);
        % For a successfully processed file, write filename to
        % 'processedFiles.txt'
        fid = fopen([dirName, '/', flockName, '/processedFiles.txt'], 'a');
        fs = strsplit(files{i}, '/');
        f = fs{end};
        fprintf(fid, '%s\n', f);
        fclose(fid);
    catch MExc
        fprintf(['Error in File ', num2str(i), ': ', files{i}, '\n'])
        disp(MExc.message)
        for k=1:length(MExc.stack)
           disp(MExc.stack(k)) 
        end
    end
    toc
end

% delete parpool
delete(gcp('nocreate'))


disp('Done processing all files')




%% OLD HYPER PARAMS
% 
% %% Set parameters for code
% stdHyperParams.doFPFiltering = 1;
% stdHyperParams.adaptiveNoise = 1;
% stdHyperParams.lambda = 0;
% stdHyperParams.simplePatternMatching = 0;
% 
% stdHyperParams.costOfNonAsDtTA = 50;
% stdHyperParams.certaintyFactor = 1;
% stdHyperParams.useAssignmentLength = 1;
% stdHyperParams.minAssignmentThreshold = 30; %35%30;
% stdHyperParams.ghostFPFilterDist = 65;
% stdHyperParams.costOfNonAsDtMA = 10;
% stdHyperParams.eucDistWeight = 1/10;%1/3;
% 
% % params regarding: process noise, i.e. how reliable the predictions of the
% % motion models are
% % params for constant acceleration model:
% stdHyperParams.posNoise = 30;%%110;%60;%50
% stdHyperParams.motNoise = 10;%;1;%5;%10
% stdHyperParams.accNoise = 1;%0.1;%1;%3
% % params for brownian motion model
% stdHyperParams.posNoiseBrownian = 80; 
% % params shared across all motion models
% stdHyperParams.quatNoise = 0.1;
% stdHyperParams.quatMotionNoise = 1; % not used, brownian motion model for rotation works best
% 
% % params regarding: measurement noise i.e. determine how reliable
% % measurements are
% stdHyperParams.measurementNoise = 50;%50
% stdHyperParams.certaintyScale = 5;%6.5
% 
% % params regarding: initialization of ghost birds
% %minimal distance for new ghost birds to other (ghost) birds that has to be free.
% stdHyperParams.minDistToBird = [95 80 50 40]; % number at index i is used when i detections are present
% % params regarding: initialization of birds from ghost bird when all others are known
% stdHyperParams.minTrustworthyness = 10;
% 
% % params regarding: initialization of birds
% stdHyperParams.initThreshold = 1;%0.85;
% stdHyperParams.initThreshold4 = 3;
% stdHyperParams.costDiff = 1.2;
% stdHyperParams.costDiff4 = 1;
% 
% stdHyperParams.initThresholdTight = 0.6;
% stdHyperParams.initThreshold4Tight = 2.5;
% stdHyperParams.costDiffTight = 2;
% stdHyperParams.costDiff4Tight = 1.5;
% stdHyperParams.patternSimilarityThreshold = 1.25;%1;
% 
% stdHyperParams.modelType = 'LieGroup';
% 
% stdHyperParams.visualizeTracking = 0;
