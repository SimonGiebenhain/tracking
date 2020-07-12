% This script allows to specify a filename to a.txt file holding the raw
% detections, a filename to the directory holding the .vsk files for the
% patterns and a filename pointing to the .csv file holding the MOT
% results.
% Then the detecitons and MOT results are visualized, in order to evaluate
% the perfromance qualitatively.


%% load patterns nd detections
dataFolder = 'multiple_object_tracking_project/datasets';
flock = 3;
recordingName = 'Starling_Trials_15-12-2019_08-30-00';
dataFilename =  [dataFolder, '/flock', num2str(flock), '/', recordingName];
patternDirectoryName = [dataFolder, '/flock', num2str(flock), '/patterns'];


fnameChunks = strsplit(recordingName, '_');
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
    % Also add folder with patterns to path of matlab!
    %[formattedData, patternsPlusNames] = readVICONcsv(dataFilename, patternDirectoryName);
    formattedData = readTxtData([dataFilename, '.txt']);
end

patternsPlusNames = read_patterns(patternDirectoryName);
patterns = zeros(length(patternsPlusNames),4,3);
patternNames = {};
for i=1:length(patternsPlusNames)
    patterns(i,:,:) = patternsPlusNames(i).pattern;
    patternNames{i} = patternsPlusNames(i).name;
end

colors = distinguishable_colors(length(patternNames));
if flock == 2
    if day >= 9
        patterns(7, :, :) = [];
        patternNames(7) = [];
        colors(7, :) = [];
    end
    if day >= 9 && hour > 12
        patterns(10, :, :) = [];
        patternNames(10) = [];
        colors(10, :) = [];
    end
elseif flock == 3
    if (day == 14 && hour == 14 && minute > 15) || ...
            (day == 14 && hour > 14) || day > 14
        patterns(12, :, :) = [];
        patternNames(12) = [];
        colors(12, :) = [];
    end
    if (day == 17 && hour == 9 && minute > 15 ) || ...
            (day == 17 && hour > 9) || day > 17
        patterns(7, :, :) = [];
        patternNames(7) = [];
        colors(7, :) = [];
    end
else
   warning(['Pattern details not implemented for flock ' , num2str(flock)]) 
end

T = find(any(~isnan(formattedData(:, :, 1)), 2), 1, 'last');
formattedData = formattedData(5000:T, :, :);

%% import csv
%fname = 'testExport.csv';
useVICONformat = 1;
[pos, quats] = importFromCSV('testExport.csv', useVICONformat);

%[pos, quats] = importFromCSV([dataFolder, '/flock', num2str(flock), '/RESULTS/', recordingName, 'RESULT.csv'], useVICONformat);
T = min(T, size(pos,2) );
pos = pos(:, 1:T, :);
quats = quats(:, 1:T, :);
%% visualize
vizParams.vizSpeed = 10;
vizParams.keepOldTrajectory = 0;
vizParams.vizHistoryLength = 500;
vizParams.startFrame = 1;
vizParams.endFrame = -1;
vizRes(formattedData, patterns, pos, quats, vizParams, 0, colors)
