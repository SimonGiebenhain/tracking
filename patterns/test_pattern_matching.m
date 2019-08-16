allPatterns = read_patterns('tracking/datasets/framework');
marker1 = allPatterns(3).pattern;
%marker1 = [1 1; 0 0; 0 -2; -2 0];
perm = randperm(size(marker1,1))
marker2 = marker1(perm, :);
pt(perm) = 1:4;
%marker2(pt(1),:) = [1000,1000,1000];
%marker2 = marker2 + 0.2*randn(size(marker2));

% m1 = [1; 0; -2; -4];
% perm = randperm(size(m1,1))
% m2 = m1(perm,:) + 0.5*randn(size(m1));

assignment = match_patterns(marker1, marker2, 'new')
p(assignment) = 1:length(assignment)

%p - perm