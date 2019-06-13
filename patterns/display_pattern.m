numbers = [4; 12;];
numbers = [15;16];
numbers = [19;20];
numbers = [1;2;4;5;8;10;12;15;16;17;19;20];

patterns = cell(length(numbers),1);
figure; hold on; grid on;
for k = 1:length(numbers)
    filename = sprintf('~/uni/7sem/project/Vicon_no_videos/nb%02d.vsk',numbers(k))
    pattern = read_pattern(filename);
    patterns{k} = pattern;
    if k <= 5
        scatter3(pattern(:,1), pattern(:,2), pattern(:,3), 'o')
    else
        scatter3(pattern(:,1), pattern(:,2), pattern(:,3), '+')
    end
end
hold off;

function pattern = read_pattern(filename)
    pattern = zeros(4,3);
    fid = fopen(filename);
    if fid == -1
       parts = strsplit(filename, '.');
       filename = [parts{1,1} 'b.' parts{1,2}]
       fid = fopen(filename);
       if fid == -1
           error('cannot open file')
       end
    end
    tline = fgetl(fid);
    idx = 1;
    while ischar(tline)
        disp(tline)
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
            if idx == 4
                break;
            end
        end
        tline = fgetl(fid);
    end
    fclose(fid);
end

