function [estPos, estQuat, certainties, ghostTracks] = birdsMOT(dataFilename, patternDirectoryName, stdHyperParams)
%BIRDSMOT Summary of this function goes here
%   Detailed explanation goes here
%% load data and patterns
% Also add folder with patterns to path of matlab!
dirPath = pwd;%'/Users/sigi/uni/7sem/project/datasets/';
dataFolder = 'multiple_object_tracking_project/datasets';
exportFolder = [dataFolder, 'RESULTS'];
filePrefix = strsplit(dataFilename, '.');
filePrefix = filePrefix{1};
if isfile([dirPath, '/', dataFolder, '/', filePrefix, '.mat'])
    load([dirPath, '/', dataFolder, '/', filePrefix, '.mat']);
else
    % Also add folder with patterns to path of matlab!
    %[formattedData, patternsPlusNames] = readVICONcsv(dataFilename, patternDirectoryName);
    formattedData = readTxtData([dirPath, '/', dataFolder, '/', dataFilename]);
end
patternsPlusNames = read_patterns([dirPath, '/', patternDirectoryName]);
patterns = zeros(length(patternsPlusNames),4,3);
patternNames = {};
for i=1:length(patternsPlusNames)
    patterns(i,:,:) = patternsPlusNames(i).pattern;
    patternNames{i} = patternsPlusNames(i).name;
end
fprintf('Loaded data successfully!\n')    
    
%% test MOT
quatMotionType = 'brownian';
fprintf('Starting to track!\n')

stdHyperParams.visualizeTracking = 0;
[estimatedPositions, estimatedQuats, snapshots, certainties, ghostTracks] = ownMOT(formattedData, patterns, patternNames ,0 , -1, size(patterns, 1), 0, -1, -1, quatMotionType, stdHyperParams);
%%
fprintf('starting Backward Track!\n')
[estimatedPositionsBackward, estimatedQuatsBackward, ~, ~] = ownMOT(formattedData, patterns, patternNames ,0 , -1, size(patterns, 1), 0, -1, -1, quatMotionType, stdHyperParams, snapshots);
revIdx = sort(1:length(formattedData), 'descend');

estimatedPositionsBackward = estimatedPositionsBackward(:, revIdx, :);
estimatedQuatsBackward = estimatedQuatsBackward(:, revIdx, :);

%% Combine forward and backward MOT results
missingFramesForwardPos = isnan(estimatedPositions);
missingFramesForwardQuat = isnan(estimatedQuats);

estPos = estimatedPositions;
estQuat = estimatedQuats;
estPos(missingFramesForwardPos) = estimatedPositionsBackward(missingFramesForwardPos);
estQuat(missingFramesForwardQuat) = estimatedQuatsBackward(missingFramesForwardQuat);

%% Export Results
if ~exist(exportFolder, 'dir')
       mkdir(exportFolder)
end

exportToCSV([dirPath, '/', exportFolder, '/', filePrefix, 'RESULT', '.csv'], estPos, estQuat, patternNames, 1)
end

