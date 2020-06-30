function exportToCSV(filename, positions, quats, patternNames, useVICONformat)
%EXPORTTOCSV Exports tracking results to .csv file
%   Arguments:
%   @filename string denoting the filename to write to
%   @positions array of dimensions [K x T x 3], where K is the number of
%   objects and T the number of frames. Contains position estimates.
%   @quats array of dimensions [K x T x 4], where K is the number of
%   objects and T the number of frames. Contains orientation estimates.
%   @patternNames cell array of length K, containing the names of patterns
%   @useVICONformat when 1 use t convention of VICON, which has the 
%   components of a quaternion q in the following order q = [qx, qy, qz,qw]
%   Otherwise the standard convention of q = [qw, qx, qy, qz] is used. The
%   latter version is used in MATLAB.
%
%   Details:
%   The written .csv file will have the following format:
%   Time (1 column) | patternName (1 column) | quaternion (4 columns) | position (3 columns)

nObjects = size(positions, 1);
T = size(positions, 2);

quats = quats ./ sqrt(sum(quats.^2, 3));

%reorder quaternions when specified
if useVICONformat == 1
    quats = cat(3, quats(:,:,2:4), quats(:,:,1));
end

%bring data in requested format
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
end


