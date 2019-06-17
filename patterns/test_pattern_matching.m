marker1 = [1 1; 0 0; 0 -2; -2 0];
perm = randperm(size(marker1,1));
perm = perm(1:3)
marker2 = marker1(perm,:);
marker2 = marker2 + 0.2*randn(size(marker2));

% m1 = [1; 0; -2; -4];
% perm = randperm(size(m1,1))
% m2 = m1(perm,:) + 0.5*randn(size(m1));

assignemnt = match_patterns(marker1, marker2);