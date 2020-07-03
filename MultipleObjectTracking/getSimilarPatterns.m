function closePairs = getSimilarPatterns(patterns, similarityThreshold)
%GETSIMILARPATTERNS Computes pairs of similar patterns
%   runs the match_patterns(., ., 'noKnowledge') algorithm in order to
%   determine patterns that are similar. Therfore all 3-subsets of the
%   patterns are compared among each other.
%   The algorithm returns all similar patterns. This information can be used
%   to enable a safe initialization of tracks, if only 3 markers are
%   detected.
%
%   Arguments:
%   @patterns array of dimensions [K x 4 x 3], where K is the number of
%   patterns, hence each pattern is a [4x3] array
%   @similarityThreshold scalar determining the threshold on the
%   mean-squared-error(MSE) between two patterns, determining what patterns are
%   similar. The MSE is computed by first solving the registration problem
%   between the two patterns (with the match_patterns() function) and then
%   applying the umeyama method.
%
%   Returns:
%   @closePairs [Mx2] array, where M is the number of similar patterns.

% Similarity is only considered for 3-subsets of the patterns.
% diff_mat_4 = zeros(length(patterns));
% for i=1:length(patterns)
%     for j=1:length(patterns)
%         pattern = squeeze(patterns(i, :, :));
%         dets = squeeze(patterns(j, :, :));
%         p = match_patterns(pattern, dets, 'noKnowledge');
% 
%         assignment = zeros(4,1);
%         assignment(p) = 1:length(p);
%         assignment = assignment(1:size(dets,1),1);
%         pattern = pattern(assignment,:);
%         pattern = pattern(assignment > 0, :);
%         [~, ~, MSE] = umeyama(pattern', dets');
%         diff_mat_4(i,j) = MSE;
%     end
% end
% diff_mat_4 = diff_mat_4 + eye(length(patterns))*100;

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
        [~, ~, MSE] = umeyama(pattern', dets');
        diff_mat_3_1(i,j) = MSE;
    end
end
diff_mat_3_1 = diff_mat_3_1 + 100*eye(length(patterns));


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
        [~, ~, MSE] = umeyama(pattern', dets');
        diff_mat_3_2(i,j) = MSE;
    end
end
diff_mat_3_2 = diff_mat_3_2 + 100*eye(length(patterns));


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
        [~, ~, MSE] = umeyama(pattern', dets');
        diff_mat_3_3(i,j) = MSE;
    end
end
diff_mat_3_3 = diff_mat_3_3 + 100*eye(length(patterns));


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
        [~, ~, MSE] = umeyama(pattern', dets');
        diff_mat_3_4(i,j) = MSE;
    end
end
diff_mat_3_4 = diff_mat_3_4 + 100*eye(length(patterns));


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