function [estPos, estQuat, certainties, ghostTracks] = birdsMOT(dataFilename, dataFolder, stdHyperParams)
%BIRDSMOT This function takes the central role in this multiple object
%tracking framwork, by directing everything from reading the input file and
%patterns, to running the MOT algorithm itself and saving the results.
%   Arguments:
%   @dataFilename string containing the name of .txt file containing all detections
%   from the VICON system, which has to be located inside the 'datasets'
%   folder.
%   @patternDirectoryName string containing the name of folder, containing 
%   all .vsk files used for the recording.
%   @stdHyperParams strcut containing relevant paramters for the MOT
%   algorithm.
%
%   Return Values:
%   @estPos array of dimensions [numObjects x numTimesteps x 3] cotaining
%   the position estimate for every object at every timestep. NaN values
%   are used, when object not detected at a time step.
%   @estQuat array of dimensions [numObjects x numTimesteps x 4] containing
%   the estimated quaternion for every object at every timestep. Again NaN
%   values are used when objects are missing.
%   @certainties not used, detaild description: TODO
%   @ghostTracks not used, detaild description: TODO


%% load data and patterns
dirPath = pwd;
% Results will be stored here
exportFolder = [dataFolder, '/RESULTS'];
filePrefix = strsplit(dataFilename, '.');
filePrefix = filePrefix{1};

fileNameChunks = strsplit(dataFilename, '/');
fname = fileNameChunks{end};
fnames = strsplit(fname, '.');
fname = fnames{1};

fnameChunks = strsplit(fname, '_');
date = fnameChunks{3};
dateChunks = strsplit(date, '-');
day = str2double(dateChunks{1});
time = fnameChunks{4};
timeChunks = strsplit(time, '-');
hour = str2double(timeChunks{1});

if isfile([filePrefix, '.mat'])
    load([filePrefix, '.mat']);
else
    % old .csv input format: [formattedData, patternsPlusNames] = readVICONcsv(dataFilename, patternDirectoryName);
    formattedData = readTxtData(dataFilename);
end
patternsPlusNames = read_patterns([dirPath, '/', dataFolder, '/patterns']);
patterns = zeros(length(patternsPlusNames),4,3);
patternNames = {};
for i=1:length(patternsPlusNames)
    patterns(i,:,:) = patternsPlusNames(i).pattern;
    patternNames{i} = patternsPlusNames(i).name;
end
fprintf('Loaded data successfully!\n')    


if day >= 9
    patterns(7, :, :) = [];
    patternNames(7) = [];
end
if day >= 9 && hour > 12
    patterns(10, :, :) = [];
    patternNames(10) = [];
end


%% Forward MOT
quatMotionType = 'brownian';
fprintf('Starting to track!\n')

stdHyperParams.visualizeTracking = 0;
[estimatedPositions, estimatedQuats, snapshots, certainties, ghostTracks] = ownMOT(formattedData, patterns, patternNames ,0 , -1, size(patterns, 1), 0, -1, -1, quatMotionType, stdHyperParams);
%% Backward MOT
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

exportToCSV([dirPath, '/', exportFolder, '/', fname, 'RESULT', '.csv'], estPos, estQuat, patternNames, 1)
end

