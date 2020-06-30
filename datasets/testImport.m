%% load patterns nd detections
dataFilename =  'multiple_object_tracking_project/datasets/Starling_Trials_10-12-2019_09-00-00.txt';%'datasets/session8/Starling_Trials_10-12-2019_16-00-00_Trajectories_100.csv'; %'datasets/Starling_Trials_10-12-2019_08-30-00.txt';%%'datasets/session8/all.csv'; %
patternDirectoryName = 'multiple_object_tracking_project/datasets/session8';
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

%% import csv
fname = 'multiple_object_tracking_project/datasets/RESULTS3/Starling_Trials_10-12-2019_09-00-00RESULT.csv';
%fname = 'testExport.csv';
useVICONformat = 1;
[pos, quats] = importFromCSV(fname, useVICONformat);
%% visualize
vizParams.vizSpeed = 10;
vizParams.keepOldTrajectory = 0;
vizParams.vizHistoryLength = 500;
vizParams.startFrame = 1;
vizParams.endFrame = -1;
% check 11.900 light blue bird again!!!
vizRes(formattedData, patterns, pos, quats, vizParams, 0)
