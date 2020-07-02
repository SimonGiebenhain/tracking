function [unprocessedFiles] = getUnprocessedFiles(dirName, processedFileNames)
%GETUNPROCESSEDFILES Get all .txt files in directory dirName, except for the files
%lsited in processedFileNames
%   Arguments:
%   @dirName string containing path to folder holding the files of interest
%   @processedFileNames name of .txt file where each line indicates a file
%   that has already been processed. This file has to contain its own
%   filename as well.
%
%   Returns:
%   @unprocessedFiles cell array, where each entry is a string of file to
%   be processed.

fileListing = dir(dirName);
if processedFileNames == -1
    processedFiles = {};
else
    fID = fopen(processedFileNames);
    processedFilesCell = textscan(fID, '%s');
    processedFiles = processedFilesCell{1};
end

unprocessedFiles = {};
idx = 1;


for i=1:length(fileListing)
    name = fileListing(i).name;
    splitName = strsplit(name, '.');
    if length(splitName)==2
       if strcmp(splitName{2}, 'txt')
           if ~any(strcmp(processedFiles, name))
               unprocessedFiles{idx} = [fileListing(i).folder, '/', name];
               idx = idx + 1;
           end
       end
    end
end

end

