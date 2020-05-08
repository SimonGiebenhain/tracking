function closePairs = getSimilarPatterns(patterns, similarityThreshold)
%GETSIMILARPATTERNS Computes pairs of similar patterns
%   runs the match_patterns(., ., 'noKnowledge') algorithm in order to
%   determine patterns that are similar. Furthermore all 3-subsets of the
%   patterns are compared among each other.
%   The algorithm return all similar patterns. This information can be used
%   to enable a safe initialization of tracks, if only 3 markers are
%   detected.

% patternDirectoryName = 'datasets/session2';
% patternsPlusNames = read_patterns(patternDirectoryName);
% patterns = zeros(length(patternsPlusNames),4,3);
% patternNames = {};
% for i=1:length(patternsPlusNames)
%     patterns(i,:,:) = patternsPlusNames(i).pattern;
%     patternNames{i} = patternsPlusNames(i).name;
% end

diff_mat_4 = zeros(length(patterns));
for i=1:length(patterns)
    for j=1:length(patterns)
        pattern = squeeze(patterns(i, :, :));
        dets = squeeze(patterns(j, :, :));
        p = match_patterns(pattern, dets, 'noKnowledge');

        assignment = zeros(4,1);
        assignment(p) = 1:length(p);
        assignment = assignment(1:size(dets,1),1);
        pattern = pattern(assignment,:);
        pattern = pattern(assignment > 0, :);
        [R, translation, MSE] = umeyama(pattern', dets');
        diff_mat_4(i,j) = MSE;
    end
end
diff_mat_4 = diff_mat_4 + eye(length(patterns))*100;

diff_mat_3_1 = zeros(length(patterns));
for i=1:length(patterns)
    for j=1:length(patterns)
        pattern = squeeze(patterns(i, :, :));
        idx = [2 3 4];
        dets = squeeze(patterns(j, idx, :));
        p = match_patterns(pattern, dets, 'noKnowledge');

        assignment = zeros(4,1);
        assignment(p) = 1:length(p);
        assignment = assignment(1:size(dets,1),1);
        pattern = pattern(assignment,:);
        pattern = pattern(assignment > 0, :);
        [R, translation, MSE] = umeyama(pattern', dets');
        diff_mat_3_1(i,j) = MSE;
    end
end
diff_mat_3_1 = diff_mat_3_1 + 100*eye(length(patterns));%triu(ones(length(patterns)))*10;


diff_mat_3_2 = zeros(length(patterns));
for i=1:length(patterns)
    for j=1:length(patterns)
        pattern = squeeze(patterns(i, :, :));
        idx = [1 3 4];
        dets = squeeze(patterns(j, idx, :));
        p = match_patterns(pattern, dets, 'noKnowledge');

        assignment = zeros(4,1);
        assignment(p) = 1:length(p);
        assignment = assignment(1:size(dets,1),1);
        pattern = pattern(assignment,:);
        pattern = pattern(assignment > 0, :);
        [R, translation, MSE] = umeyama(pattern', dets');
        diff_mat_3_2(i,j) = MSE;
    end
end
diff_mat_3_2 = diff_mat_3_2 + 100*eye(length(patterns));%triu(ones(length(patterns)))*10;


diff_mat_3_3 = zeros(length(patterns));
for i=1:length(patterns)
    for j=1:length(patterns)
        pattern = squeeze(patterns(i, :, :));
        idx = [1 2 4];
        dets = squeeze(patterns(j, idx, :));
        p = match_patterns(pattern, dets, 'noKnowledge');

        assignment = zeros(4,1);
        assignment(p) = 1:length(p);
        assignment = assignment(1:size(dets,1),1);
        pattern = pattern(assignment,:);
        pattern = pattern(assignment > 0, :);
        [R, translation, MSE] = umeyama(pattern', dets');
        diff_mat_3_3(i,j) = MSE;
    end
end
diff_mat_3_3 = diff_mat_3_3 + 100*eye(length(patterns));%triu(ones(length(patterns)))*10;


diff_mat_3_4 = zeros(length(patterns));
for i=1:length(patterns)
    for j=1:length(patterns)
        pattern = squeeze(patterns(i, :, :));
        idx = [1 2 3];
        dets = squeeze(patterns(j, idx, :));
        p = match_patterns(pattern, dets, 'noKnowledge');

        assignment = zeros(4,1);
        assignment(p) = 1:length(p);
        assignment = assignment(1:size(dets,1),1);
        pattern = pattern(assignment,:);
        pattern = pattern(assignment > 0, :);
        [R, translation, MSE] = umeyama(pattern', dets');
        diff_mat_3_4(i,j) = MSE;
    end
end
diff_mat_3_4 = diff_mat_3_4 + 100*eye(length(patterns));%triu(ones(length(patterns)))*10;



% min(diff_mat_4, [], 2)
% min(diff_mat_3_1, [], 2)
% min(diff_mat_3_2, [], 2)
% min(diff_mat_3_3, [], 2)
% min(diff_mat_3_4, [], 2)

closePairs = zeros(0,2);
[rows_3_1, cols_3_1] = find(diff_mat_3_1 < similarityThreshold);
closePairs = [closePairs; [rows_3_1 cols_3_1]];

[rows_3_2, cols_3_2] = find(diff_mat_3_2 < similarityThreshold);
closePairs = [closePairs; [rows_3_2 cols_3_2]];

[rows_3_3, cols_3_3] = find(diff_mat_3_3 < similarityThreshold);
closePairs = [closePairs; [rows_3_3 cols_3_3]];

[rows_3_4, cols_3_4] = find(diff_mat_3_4 < similarityThreshold);
closePairs = [closePairs; [rows_3_4 cols_3_4]];

closePairs = unique(closePairs, 'rows');