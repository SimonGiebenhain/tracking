function [estPos, estQuat, certainties, ghostTracks] = birdsMOT(dataFilename, patternDirectoryName, stdHyperParams)
%BIRDSMOT Summary of this function goes here
%   Detailed explanation goes here
%% load data and patterns
% Also add folder with patterns to path of matlab!
filePrefix = strsplit(dataFilename, '.');
filePrefix = filePrefix{1};
if isfile([filePrefix, '.mat'])
    load([filePrefix, '.mat']);
else
    % Also add folder with patterns to path of matlab!
    %[formattedData, patternsPlusNames] = readVICONcsv(dataFilename, patternDirectoryName);
    formattedData = readTxtData(dataFilename);
end
patternsPlusNames = read_patterns(patternDirectoryName);
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
[estimatedPositions, estimatedQuats, snapshots, certainties, ghostTracks] = ownMOT(formattedData(beginningFrame:endFrame,:,:), patterns, patternNames ,0 , -1, size(patterns, 1), 0, -1, -1, quatMotionType, stdHyperParams);
%%
[estimatedPositionsBackward, estimatedQuatsBackward, ~, ~] = ownMOT(formattedData(beginningFrame:endFrame,:,:), patterns, patternNames ,0 , -1, size(patterns, 1), 0, -1, -1, quatMotionType, stdHyperParams, snapshots);
revIdx = sort(1:endFrame, 'descend');

estimatedPositionsBackward = estimatedPositionsBackward(:, revIdx, :);
estimatedQuatsBackward = estimatedQuatsBackward(:, revIdx, :);

%% Combine forward and backward MOT results
missingFramesForwardPos = isnan(estimatedPositions);
missingFramesForwardQuat = isnan(estimatedQuats);

estPos = estimatedPositions;
estQuat = estimatedQuats;
estPos(missingFramesForwardPos) = estimatedPositionsBackward(missingFramesForwardPos);
estQuat(missingFramesForwardQuat) = estimatedQuatsBackward(missingFramesForwardQuat);
end

