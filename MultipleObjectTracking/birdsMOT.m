function [estPos, estQuat, formattedData, patterns, patternNames, certainties, ghostTracks] = birdsMOT(dataFilename, dataFolder, stdHyperParams, flock, beginningFrame, endFrame)
%BIRDSMOT This function takes the central role in this multiple object
%tracking framwork, by directing everything from reading the input file and
%patterns, to running the MOT algorithm itself and saving the results.
%   Arguments:
%   @dataFilename string containing the name of .txt file containing all detections
%   from the VICON system, which has to be located inside the 'datasets'
%   folder. Note that the file extension is not passed.
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

fileNameChunks = strsplit(dataFilename, '/');
fname = fileNameChunks{end};

fnameChunks = strsplit(fname, '_');
date = fnameChunks{3};
dateChunks = strsplit(date, '-');
day = str2double(dateChunks{1});
time = fnameChunks{4};
timeChunks = strsplit(time, '-');
hour = str2double(timeChunks{1});
minute = str2double(timeChunks{2});

if isfile([dataFilename, '.mat'])
    load([dataFilename, '.mat']);
else
    % old .csv input format: [formattedData, patternsPlusNames] = readVICONcsv(dataFilename, patternDirectoryName);
    formattedData = readTxtData([dataFilename, '.txt']);
end
patternsPlusNames = read_patterns([dirPath, '/', dataFolder, '/patterns']);
patterns = zeros(length(patternsPlusNames),4,3);
patternNames = {};
for i=1:length(patternsPlusNames)
    patterns(i,:,:) = patternsPlusNames(i).pattern;
    patternNames{i} = patternsPlusNames(i).name;
end
fprintf('Loaded data successfully!\n')    

colors = distinguishable_colors(size(patterns,1));
if flock == 2
    if (day == 8 && hour == 9 && minute > 15) || ...
            (day == 8 && hour > 9) || day > 8
        patterns(7, :, :) = [];
        patternNames(7) = [];
        colors(7, :) = [];
    end
    if (day == 9 && hour == 8 && minute > 15 ) || ...
            (day == 9 && hour > 8) || day > 9
        patterns(10, :, :) = [];
        patternNames(10) = [];
        colors(10, :) = [];
    end
elseif flock == 3 %ATTENTION: not done yet, only first estimate!
    if (day == 14 && hour == 15 && minute > 15) || ...
            (day == 14 && hour > 15) || day > 14
        patterns(12, :, :) = [];
        patternNames(12) = [];
        colors(12, :) = [];
    end
    if (day == 17 && hour == 10 && minute > 0 ) || ...
            (day == 17 && hour > 10) || day > 17
        patterns(6, :, :) = [];
        patternNames(6) = [];
        colors(6, :) = [];
    end
else
    warning(['details for flock ' num2str(flock), 'not yet implemented']) 
end

T = find(any(~isnan(squeeze(formattedData(:, :, 1))), 2), 1, 'last');

if exist('beginningFrame', 'var')
    if endFrame == -1
        endFrame = T;
    end
   formattedData = formattedData(beginningFrame:endFrame, :, :); 
else
   formattedData =  formattedData(1:T, :, :); 
end

%% Forward MOT
quatMotionType = 'brownian';
fprintf('Starting to track!\n')

[estimatedPositions, estimatedQuats, snapshots, certainties, ghostTracks] = ownMOT(formattedData, patterns, patternNames ,0 , -1, size(patterns, 1), 0, -1, -1, quatMotionType, stdHyperParams, colors);
%% Backward MOT
fprintf('starting Backward Track!\n')
[estimatedPositionsBackward, estimatedQuatsBackward, ~, ~] = ownMOT(formattedData, patterns, patternNames ,0 , -1, size(patterns, 1), 0, -1, -1, quatMotionType, stdHyperParams, colors, snapshots, estimatedPositions, estimatedQuats);
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

T = exportToCSV([dirPath, '/', exportFolder, '/', fname, 'RESULT', '.csv'], estPos, estQuat, patternNames, 1);

fid = fopen([dataFolder, '/recordingLengths.txt'], 'a');
fprintf(fid, '%s\n', [fname , ': ', num2str(T)]);
fclose(fid);
end

