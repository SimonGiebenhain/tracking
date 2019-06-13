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