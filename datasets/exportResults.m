%% load Data
load('trackingResults')
nObjects = size(estimatedPositions, 1);
T = size(estimatedPositions, 2);

%% rearange data

% normalize estimated quats
estimatedQuats = estimatedQuats ./ sqrt(sum(estimatedQuats.^2, 3));
% reorder quats
estimatedQuats = cat(3, estimatedQuats(:,:,2:4), estimatedQuats(:,:,1));
%estimatedQuats(:,:,1:3) = estimatedQuats(:,:,2:4);
%estimatedQuats(:,:,4) = estimatedQuats(:,:,1);

completeTable = zeros(T, nObjects*(3+4));
for k=1:nObjects
    completeTable(:,(k-1)*7+1:k*7) = [ squeeze(estimatedQuats(k,:,:)) squeeze(estimatedPositions(k,:,:))];
end

% prepend frame column
frame1 = 1011 + 130;
finalTable = zeros(size(completeTable,1), size(completeTable,2)+1);
finalTable(:,2:end) = completeTable;
finalTable(:,1) = frame1:frame1+T-1;



%% write to csv
fnam = 'kalmanFilterPredictions.csv';
allPatterns = read_patterns('tracking/datasets/framework');
header1 = cell(1,nObjects*7+1);
header1{1,1} = '';
for k=1:nObjects
    tmp = strsplit(allPatterns(k).name, '.');
    patternName = tmp{1};
    header1{1,(k-1)*7+2} = patternName;
end

header2 = cell(1,nObjects*7+1);
header2{1,1} = 'Frame';
for k=1:nObjects
    header2{1,(k-1)*7+2} = 'RX';
    header2{1,(k-1)*7+3} = 'RY';
    header2{1,(k-1)*7+4} = 'RZ';
    header2{1,(k-1)*7+5} = 'RW';
    header2{1,(k-1)*7+6} = 'TX';
    header2{1,(k-1)*7+7} = 'TY';
    header2{1,(k-1)*7+8} = 'TZ';
end

fmt = repmat('%s, ', 1, length(header1));
fmt(end:end+1) = '\n';
fid = fopen(fnam, 'w');
fprintf(fid, fmt, header1{:});
fprintf(fid, fmt, header2{:});
fclose(fid);
dlmwrite(fnam,finalTable,'-append','delimiter',',');
