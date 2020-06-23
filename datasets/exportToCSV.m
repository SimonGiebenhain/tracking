function exportToCSV(filename, positions, quats, patternNames, useVICONformat)
%EXPORTTOCSV Summary of this function goes here
%   Detailed explanation goes here

nObjects = size(positions, 1);
T = size(positions, 2);

quats = quats ./ sqrt(sum(quats.^2, 3));

if useVICONformat == 1
    quats = cat(3, quats(:,:,2:4), quats(:,:,1));
end

completeTable = cell(T*nObjects, 1+1+4+3);

for t=1:T
    for k=1:nObjects
        completeTable{(t-1)*nObjects+k, 1} = t;
        completeTable{(t-1)*nObjects+k, 2} = patternNames{k};
        
        completeTable{(t-1)*nObjects+k, 3} = quats(k,t, 1);
        completeTable{(t-1)*nObjects+k, 4} = quats(k,t, 2);
        completeTable{(t-1)*nObjects+k, 5} = quats(k,t, 3);
        completeTable{(t-1)*nObjects+k, 6} = quats(k,t, 4);

        completeTable{(t-1)*nObjects+k, 7} = positions(k,t,1);
        completeTable{(t-1)*nObjects+k, 8} = positions(k,t,2);
        completeTable{(t-1)*nObjects+k, 9} = positions(k,t,3);

    end
end

%write to csv file
header = cell(1, 9);
header{1,1} = 'Frame';
header{1,2} = 'patternID';
header{1,3} = 'RX';
header{1,4} = 'RY';
header{1,5} = 'RZ';
header{1,6} = 'RW';
header{1,7} = 'X';
header{1,8} = 'Y';
header{1,9} = 'Z';

T = cell2table(completeTable,'VariableNames', header);
writetable(T, filename);
%fmt = repmat('%s, ', 1, length(header));
%fmt(end:end+1) = '\n';
%fid = fopen(filename, 'w');
%fprintf(fid, fmt, header{:});
%fclose(fid);
%dlmwrite(filename, completeTable, '-append', 'delimiter', ',');
end


