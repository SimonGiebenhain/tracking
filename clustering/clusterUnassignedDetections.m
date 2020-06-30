function clusters = clusterUnassignedDetections(detections, epsilon, debug)
%CLUSTER_UNASSIGNEDDETECTIONS clusters points using a hierachical
%agglomorative clustering approach.
%   @detections array of dimensions [nx3], where n is the number of points.
%   @epsilon points that are further apart than epsilon will not be in the
%   same cluster, if they are close than epsilon, they will be in the same
%   clsuter.
%   @debug variable for depug purposes, when set the method will display a
%   dendrogram.
%
%   Return values:
%   @clusters cell array where the i-th cell is an array of dimensions [n_i x 3],
%   where n_i is the number of points in that cluster

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


