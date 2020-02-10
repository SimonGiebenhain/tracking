function [outputArg1,outputArg2] = exportToCSV(filename, positions, quats, frame1, patternNames, useVICONformat, includeRMSE, viconPos, viconQuat)
%EXPORTTOCSV Summary of this function goes here
%   Detailed explanation goes here

nObjects = size(positions, 1);
T = size(positions, 2);


% if includeRMSE
%     offset = 130;
%     % load VICON predictions
%     load('D_labeled.mat')
%     D_offset = D_labeled(offset:end,:);
%     VICONX = D_offset(:,5:7:end);
%     VICONY = D_offset(:,6:7:end);
%     VICONZ = D_offset(:,7:7:end);
%     VICONq2 = D_offset(:,1:7:end);
%     VICONq3 = D_offset(:,2:7:end);
%     VICONq4 = D_offset(:,3:7:end);
%     VICONq1 = D_offset(:,4:7:end);
% 
%     VICONPos = cat(3, VICONX, VICONY, VICONZ);
%     VICONPos = permute(VICONPos, [2 1 3]);
% 
%     VICONquat = cat(3, VICONq1, VICONq2, VICONq3, VICONq4);
%     VICONquat = permute(VICONquat, [2 1 3]);
% end

quats = quats ./ sqrt(sum(quats.^2, 3));

if includeRMSE
    % calculate pose error to VICON prediictions
    poseError = NaN*zeros(nObjects, T);
    for k=1:nObjects
        for t=1:T
            myQ = squeeze(quats(k,t,:));
            myP = squeeze(positions(k,t,:));
            VICONQ = squeeze(VICONquat(k,t,:));
            VICONP = squeeze(VICONPos(k,t,:));
            pattern = squeeze(patterns(k,:,:));
            poseError(k,t) = sqrt(sum(( (Rot(myQ)*pattern')'+ myP' - (Rot(VICONQ)*pattern')' - VICONP' ).^2, 'all')/4);
        end
    end
end

if useVICONformat
    quats = cat(3, quats(:,:,2:4), quats(:,:,1));
end

if includeRMSE
    completeTable = zeros(T, nObjects*(3+4+1));
else
    completeTable = zeros(T, nObjects*(3+4));
end

for k=1:nObjects
    if includeRMSE
        completeTable(:,(k-1)*8+1:k*8) = [ squeeze(quats(k,:,:)) squeeze(positions(k,:,:)) poseError(k,:)'];
    else
        completeTable(:,(k-1)*7+1:k*7) = [squeeze(quats(k,:,:)) squeeze(positions(k,:,:))];
    end
end

% prepend frame column
finalTable = zeros(size(completeTable,1), size(completeTable,2)+1);
finalTable(:,2:end) = completeTable;
finalTable(:,1) = frame1:frame1+T-1;

%write to csv file
writeTOCSV(finalTable, filename, patternNames, includeRMSE)
end


function writeTOCSV(data, filename, patternNames, includeRMSE)
    nObjects = length(patternNames);

    if includeRMSE
        dimPerBird = 8;
    else
        dimPerBird = 7;
    end
    assert(nObjects*dimPerBird+1 == size(data,2))
    header1 = cell(1,nObjects*dimPerBird+1);
    header1{1,1} = '';
    for k=1:nObjects
        tmp = strsplit(patternNames{k}, '.');
        patternName = tmp{1};
        header1{1,(k-1)*dimPerBird+2} = patternName;
    end

    header2 = cell(1,nObjects*dimPerBird+1);
    header2{1,1} = 'Frame';
    for k=1:nObjects
        header2{1,(k-1)*dimPerBird+2} = 'RX';
        header2{1,(k-1)*dimPerBird+3} = 'RY';
        header2{1,(k-1)*dimPerBird+4} = 'RZ';
        header2{1,(k-1)*dimPerBird+5} = 'RW';
        header2{1,(k-1)*dimPerBird+6} = 'TX';
        header2{1,(k-1)*dimPerBird+7} = 'TY';
        header2{1,(k-1)*dimPerBird+8} = 'TZ';
        if includeRMSE
            header2{1,(k-1)*dimPerBird+9} = 'RMSE';
        end

    end

    fmt = repmat('%s, ', 1, length(header1));
    fmt(end:end+1) = '\n';
    fid = fopen(filename, 'w');
    fprintf(fid, fmt, header1{:});
    fprintf(fid, fmt, header2{:});
    fclose(fid);
    dlmwrite(filename, data, '-append', 'delimiter', ',');
end


