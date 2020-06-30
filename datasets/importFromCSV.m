function [pos, quats] = importFromCSV(fname, useVICONformat)
%IMPORTFROMCSV Import position and orientation estimates, previously
%written to a .csv file specified by filename
%   Arguements:
%   @fname filename of the .csv file containing the position and
%   orientation estimates. This function assumes that the .csv file follows
%   the format generated when using the EXPORTTOCSV function.
%   @usVICONformat determines the order of the components of the
%   quaternions as described in EXPORTTOCSV.
%
%   Returns:
%   @pos array of dimensions [K x T x 3] containing position estimates.
%   Here K is the number of objects and T is the number of frames.
%   @quats array of dimensions [K x T x 4] containing orientation estimates.

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

if useVICONformat == 1
    quats = cat(3, quats(:, :, 4), quats(:, :, 1:3));
end
end

