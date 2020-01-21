function [formattedData, patternsPlusNames] = readVICONcsv(file, patternDirectory)
%READVICONCSV Summary of this function goes here
%   Detailed explanation goes here

[ ~, cmdRes] = system(['wc -l ', file]);
[startIdx, endIdx] = regexp(cmdRes, '\d+\s');
nLines = str2double(cmdRes(startIdx:endIdx-1));
fprintf(['Reading csv file with ', int2str(nLines), ' lines of data.\n'])
fid = fopen(file);

splittedName = strsplit(file, '.');
storageName = [splittedName{1}, '.m'];
fprintf(['The formatted data will be stored at ', storageName, '\n'])

nColumns = 500;
data = NaN * ones(nLines-5, nColumns);

%% Load Patterns First
patternsPlusNames = read_patterns(patternDirectory);
fprintf('Done loading the patterns! \n')

%% Load VICON detections


%waste the first 5 lines
for i=1:5
    tline = fgetl(fid);
end

tline = fgetl(fid);
rowIdx = 1;
percentDone = 0.1;
while ischar(tline)
    if rowIdx/nLines > percentDone
        fprintf([num2str(percentDone*100), 'percent done!\n'])
        percentDone = percentDone + 0.1;
    end
    line = split(tline,',');
    colIdx = 1;
    for i=3:length(line)
        if ~isempty(line{i}) && ~strcmp(line{i}, 'NaN')
            val = str2double(line{i});
            data(rowIdx,colIdx) = val;
            colIdx = colIdx + 1;
        end
        if colIdx > nColumns
            fprintf('more columns than expected\n')
        end
    end
    % line for next iteration
    tline = fgetl(fid);
    rowIdx = rowIdx + 1;
    if rowIdx > size(data,1)
        fprintf('more rows than expected\n')
    end
end
fclose(fid);


% shrink data to appropriate size
empty_cols = all(isnan(data), 1);
empty_rows = all(isnan(data), 2);

data_tight = data(:, ~empty_cols);
data_tight = data_tight(~empty_rows, :);
data = data_tight;


nums = zeros(size(data,1),1);
for i=1:size(data,1)
    numDetections = sum(~isnan(data(i,:)));
    nums(i) = numDetections;
    if mod(numDetections,3) ~= 0
        fprintf('smth wrong in line %d', i)
    end
end
max(nums)

%
X = data(:,1:3:end);
Y = data(:,2:3:end);
Z = data(:,3:3:end);

formattedData = cat(3, X,Y,Z);

save(storageName, 'formattedData', 'patternsPlusNames');

end

