% There are 10 bird
% Each has 4 markers
% so assume that 4*10*3 should be enough columns
% add error margin, do 150 instead of 120

fid = fopen('tracking/datasets/session2/all_session2.csv');

data = NaN * ones(33280,300);


%waste the first 5 lines
for i=1:5
    tline = fgetl(fid)
end
%%
tline = fgetl(fid);
rowIdx = 1;
 while ischar(tline)
     rowIdx
     line = split(tline,',');
     colIdx = 1;
     for i=3:length(line)
         if ~isempty(line{i}) && ~strcmp(line{i}, 'NaN')
            val = str2double(line{i});
            data(rowIdx,colIdx) = val;
            colIdx = colIdx + 1;
         end
         if colIdx > 150
             fprintf('more columns than expected')
         end
     end
     % line for next iteration
     tline = fgetl(fid);
     rowIdx = rowIdx + 1;
     if rowIdx > size(data,1)
         fprintf('more rows than expected')
     end
 end
fclose(fid);

%%
nums = zeros(size(data,1),1);
for i=1:size(data,1)
    numDetections = sum(~isnan(data(i,:)));
    nums(i) = numDetections;
    if mod(numDetections,3) ~= 0
        fprintf('smth wrong in line %d', i)
    end
end
max(nums)

%%
X = data(:,1:3:end);
Y = data(:,2:3:end);
Z = data(:,3:3:end);

formattedData = cat(3, X,Y,Z);