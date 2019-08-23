function clusters = clusterUnassignedDetections(detections, epsilon, debug)
%CLUSTER_UNLABELED Summary of this function goes here
%   Detailed explanation goes here

if ~exist('debug', 'var')
    debug = 0;
end

dist = pdist(detections);
dendro = linkage(dist);
inconsistent(dendro);
if debug
    figure
    dendrogram(dendro)
end
% Cluster by cutting off according to distance to next points
clusterAssignment = cluster(dendro, 'cutoff', epsilon, 'Criterion', 'distance');
nClusters = max(clusterAssignment);

for c = 1:nClusters
    clusters{c} = detections(clusterAssignment == c,:);
end

end


