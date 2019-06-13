function [patterns] = read_patterns(directory_name)
%READ_PATTERNS Read patterns from '*.vsk' files in given directory
%
%   patterns = READ_PATTERNS(directory_name) returns a structured array.
%   Each element is a struct with fields 'pattern' and 'name'.
%
%   ATTENTION: This method currently assumes that all patterns have 4
%   points!
%
%   So far I ignore preallocation since there will be a limited number of
%   patterns.

num_points_per_pattern = 4;

folder = dir(directory_name);
k = 1;
for i = 1:size(folder,1)
    filename = folder(i).name;
    if contains(filename, '.vsk')
        patterns(k).pattern = read_pattern(filename);
        patterns(k).name = filename;
        k = k + 1;
    else
        continue;
    end
end
    function pattern = read_pattern(filename)
        pattern = zeros(num_points_per_pattern,3);
        fid = fopen(filename);
        if fid == -1
            parts = strsplit(filename, '.');
            filename = [parts{1,1} 'b.' parts{1,2}];
            fid = fopen(filename);
            if fid == -1
                error('cannot open file')
            end
        end
        tline = fgetl(fid);
        idx = 1;
        while ischar(tline)
            parts = strsplit(tline, '"');
            dim = 0;
            if contains(tline, '_x')
                dim = 1;
            elseif contains(tline, '_y')
                dim = 2;
            elseif contains(tline, '_z')
                dim = 3;
            else
                tline = fgetl(fid);
                continue;
            end
            pattern(idx,dim) = str2double(parts{1,4});
            if dim == 3
                idx = idx + 1;
                if idx == num_points_per_pattern + 1
                    break;
                end
            end
            tline = fgetl(fid);
        end
        fclose(fid);
    end
end


