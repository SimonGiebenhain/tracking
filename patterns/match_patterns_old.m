%%%%%%%%%
% The goal of this file is to match unassigned detections to the patterns
%%%%%%%%

%% Load detections, which could not have been assigned to pattern
load('datasets/D_unlabeled.mat')


%% Cluster unassigned detections
% Cluster unassigned detections under the assumption that distances within
% a pattern are smaller, than distances between points of different
% patterns
clusters = cluster_unlabeled(D_unlabeled, 100, 10);

%% Read Patterns
patterns = read_patterns('datasets/framework');

%% Build scoring model for matching

%function scores = match_pattern(pattern, clusters)
    
%end

%% Tests
I = [1,4,6,7];
I = 6
for i=1:length(I)
    test = clusters{1,I(i)};
    m = mean(test);
    test = test - m;
    fprintf('cluster %d', I(i))
    sort(pdist(test))
end

plot3(test(:,1), test(:,2), test(:,3), '+', 'MarkerSize', 10)
hold on; grid on;
I = [1,2,5,6];
for i = 1:length(I)
    p = patterns(I(i)).pattern;
    patterns(I(i)).name
    m = mean(p);
    p = p-m;
    sort(pdist(p))
    scatter3(p(:,1), p(:,2), p(:,3))
end
