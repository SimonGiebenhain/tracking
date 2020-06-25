function [unprocessedFiles] = getUnprocessedFiles(dirName, processedFileName)
%GETUNPROCESSEDFILES Summary of this function goes here
%   Detailed explanation goes here

fileListing = dir(dirName);
fID = fopen(processedFileName);
processedFilesCell = textscan(fID, '%s');
processedFiles = processedFilesCell{1};

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

