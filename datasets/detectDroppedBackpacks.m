%if batchStartupOptionUsed
%    cd ../..
%end
%path = pwd;
%addpath(genpath([path, '/', 'multiple_object_tracking_project']))

dirName = 'multiple_object_tracking_project/datasets/flock2';
% This file should contain the names of all .txt files that already have been processed 
files = getUnprocessedFiles(dirName, -1);

 numDets = cell(length(files), 2);

for i=1:length(files)
    fname = files{i};
    filePrefix = strsplit(fname, '.');
    filePrefix = filePrefix{1};
    if isfile([filePrefix, '.mat'])
        load([filePrefix, '.mat']);
    else
        % old .csv input format: [formattedData, patternsPlusNames] = readVICONcsv(dataFilename, patternDirectoryName);
        formattedData = readTxtData(fname);
    end
    numFrames = sum(any(~isnan(squeeze(formattedData(:, :, 1))), 2));
    numDets{i,2} = sum(~isnan(formattedData), 'all') / (3*numFrames);
    numDets{i,1} = fname;
end

disp(numDets)