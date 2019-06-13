function [data_clustered] = cluster_unlabeled(data, epsilon, max_clusters, debug)
%CLUSTER_UNLABELED Summary of this function goes here
%   Detailed explanation goes here

if ~exist('debug', 'var')
    debug = 0;
end

[T,~] = size(data);
data_clustered = cell(T,max_clusters);
for t = 1:T
    frame = data(t,:);
    % ignore single points
    % TODO also include them in clustering
    if length(frame(~isnan(frame))) < 6
        continue
    end
    frame = frame(~isnan(frame));
    frame = reshape(frame, [],1);
    N = size(frame,1);
    frame3D = cat(2, frame(1:3:N), frame(2:3:N), frame(3:3:N));
    d = pdist(frame3D);
    dendro = linkage(d);
    inconsistent(dendro);
    if debug
        figure
        dendrogram(dendro)
    end
    % Cluster by cutting off according to distance to next points
    clusters = cluster(dendro, 'cutoff', epsilon, 'Criterion', 'distance');
    num_clusters = max(clusters);
    
    for c = 1:num_clusters
        data_clustered{t,c} = frame3D(clusters == c,:);
    end
    
    %figure; hold on; grid on;
    %for c = 1:num_clusters
    %   plot3(data_clustered{t,c}(:,1), data_clustered{t,c}(:,2), data_clustered{t,c}(:,3), 'o')
    %end
    %hold off;
    
    % Checke in Vergangenheit und Zukunft, i.e. temporal vicinity, ob Vogel
    % in Nähe
    
    % Wenn nicht verwerfe cluster, aber Zeige ihn deutlich sichtbar an
    
    % Wenn ja, mache cluster zu detection von geleichem vogel
end
end

