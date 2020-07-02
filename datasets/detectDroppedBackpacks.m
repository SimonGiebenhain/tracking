if batchStartupOptionUsed
    cd ../..
end
path = pwd;
addpath(genpath([path, '/', 'multiple_object_tracking_project']))

dirName = 'multiple_object_tracking_project/datasets/flock2';
% This file should contain the names of all .txt files that already have been processed 
files = getUnprocessedFiles(dirName, -1);

numDets = zeros(length(files), 1);
parpool(64)

parfor i=1:length(files)
    fname = files{i};
    filePrefix = strsplit(fname, '.');
    filePrefix = filePrefix{1};
    if isfile([filePrefix, '.mat'])
        formattedData = load([filePrefix, '.mat']);
    else
        % old .csv input format: [formattedData, patternsPlusNames] = readVICONcsv(dataFilename, patternDirectoryName);
        formattedData = readTxtData(fname);
    end
    numFrames = sum(any(~isnan(squeeze(formattedData(:, :, 1))), 2));
    numDets(i) = sum(~isnan(formattedData), 'all') / (3*numFrames);
end

numDetsAndNames = cell{length(files), 2};
for i=1:length(files)
   numDetsAndNames{i, 1} = files{i};
   numDetsAndNames{i, 2} = numDets(i);
end

save('numberOfDetections.mat', numDetsAndNames)