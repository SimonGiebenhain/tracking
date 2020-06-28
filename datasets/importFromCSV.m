function [pos, quats] = importFromCSV(fname)
%IMPORTFROMCSV Summary of this function goes here
%   Detailed explanation goes here

workingDir = pwd;
table = readtable([workingDir, '/', fname]);
T = max(table2array(table(:, 1)));
patternIDs = table2cell(table(:, 2));
patternNames = unique(patternIDs);
numObjects = length(patternNames);
pos = NaN*zeros(numObjects, T, 3);
quats = NaN*zeros(numObjects, T, 4);

for k=1:numObjects
    pattern = patternNames{k};
    strsFound =  strfind(patternIDs, pattern);
    objIdx = find(~cellfun(@isempty, strsFound));
    pos(k, :, :) = table2array(table(objIdx, 7:9));
    quats(k, :, :) = table2array(table(objIdx, 3:6));
end

quats = cat(3, quats(:, :, 4), quats(:, :, 1:3));
end

