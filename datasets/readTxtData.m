function formattedData = readTxtData(file)
%READTXTDATA Summary of this function goes here
%   Detailed explanation goes here

nLines = 90000;
fprintf(['Starting to process ', file, ' file.\n'])
fid = fopen(file);
if fid < 0
 fprintf(2, 'failed to open "%s" because "%s"\n', file, message);
 %and here, get out gracefully
end

splittedName = strsplit(file, '.');
storageName = [splittedName{1}, '.mat'];
fprintf(['The formatted data will be stored at ', storageName, '\n'])

nColumns = 50;
data = NaN * ones(nLines, nColumns, 3);

%% Load VICON detections
tline = fgetl(fid);
percentDone = 0.1;
firstFrameFound = 0;
firstFrame = -1;
t = -1;
while ischar(tline)
    if t/nLines > percentDone
        fprintf([num2str(percentDone*100), 'percent done!\n'])
        percentDone = percentDone + 0.1;
    end
    [startIdx, endIdx] = regexp(tline, 'Frame Number: \d*');
    if ~isempty(startIdx) && startIdx == 1
        t = str2num(tline(startIdx+14:endIdx));
        if firstFrame >= 0 && t+1 == firstFrame
            break
        end
    elseif ~isempty(regexp(tline, '\s*Unlabeled Markers (', 'once'))
        [startIdx, endIdx] = regexp(tline, '\d*');
        numDets = str2num(tline(startIdx:endIdx)); 
        for i=1:numDets
            tline = fgetl(fid);
            [startIdx, endIdx] = regexp(tline, '\(.*\)');
            allDets = tline(startIdx+1:endIdx-1);
            dets = split(allDets, ',');
            data(t+1, i, 1) = str2double(dets{1});
            data(t+1, i, 2) = str2double(dets{2});
            data(t+1, i, 3) = str2double(dets{3});
        end
        if firstFrameFound == 0 && numDets > 0 
           firstFrame = t+1;
           firstFrameFound = 1;
        end
        
    end    
    % line for next iteration
    tline = fgetl(fid);
end
fclose(fid);


% shrink data to appropriate size
empty_cols = all(isnan(squeeze(data(:, :, 1))), 1);


formattedData = data(:, ~empty_cols, :);

save(storageName, 'formattedData');


end

