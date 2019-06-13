%% Load Data
%%%%% Manually %%%%%
%readtable('../datasets/20190124_10BirdsWeightTrials05_testdata.csv', 'ReadRowNames', true, 'HeaderLines', 1);
%% Discard rotation and separate X,Y,Z coordinates
[N,C] = size(D_labeled);
K = C / 7;
X = zeros(N, K);
Y = zeros(N, K);
Z = zeros(N, K);
column_idx = 1;
for c = 1:70
    if mod(c,7) == 5
        X(:,column_idx) = D_labeled(:,c); 
    elseif mod(c,7) == 6
        Y(:,column_idx) = D_labeled(:,c); 
    elseif mod(c,7) == 0
        Z(:,column_idx) = D_labeled(:,c); 
        column_idx = column_idx + 1;
    end
end

D = cat(3, X, Y, Z);

% Get extreme coordinates
X_min = min(X, [], 'all');
X_max = max(X, [], 'all');
Y_min = min(Y, [], 'all');
Y_max = max(Y, [], 'all');
Z_min = min(Z, [], 'all');
Z_max = max(Z, [], 'all');
[N,C] = size(D_unlabeled);

notnan = ~isnan(D(:,:,1));
complete_frame = prod(notnan, 2);
% t0 is the frame idx for the first frame without missed detections
t0 = 1;
while ~complete_frame(t0)
   t0 = t0 + 1; 
end

%%
figure
scatter3([X_max, X_min], [Y_max, Y_min], [Z_max, Z_min]);
hold on

p_unlabeled = plot3(D_unlabeled(t0+1,1:3:C),D_unlabeled(t0+1,2:3:C),D_unlabeled(t0+1,3:3:C),'o','MarkerFaceColor','green');
cluster_centers = process_clusters(D_clustered(t0+1,:));
%augmented_frame = assign_clusters(cluster_centers, squeeze(D(t0+1,:,:)), squeeze(D(t0,:,:)), 250);
if size(cluster_centers,1) > 0
    p_clusters = plot3(cluster_centers(:,1), cluster_centers(:,2), cluster_centers(:,3), 'o', 'MarkerEdgeColor','blue', 'MarkerSize',20);
else
    p_clusters = plot3(NaN, NaN, NaN, 'o','MarkerEdgeColor','blue', 'MarkerSize',20);
end
p_labeled = plot3(X(t0+1,:), Y(t0+1,:), Z(t0+1,:), '+', 'MarkerEdgeColor','red', 'MarkerSize', 10);
%p_labeled = plot3(augmented_frame(:,1), augmented_frame(:,2), augmented_frame(:,3), '+', 'MarkerEdgeColor','red', 'MarkerSize', 10);
grid on;
hold off
axis manual
%%
for k = t0+2:N
    cluster_centers = process_clusters(D_clustered(k,:));
    %augmented_frame = assign_clusters(cluster_centers, squeeze(D(k,:,:)), augmented_frame, 20000);
    if size(cluster_centers,1) > 0
        p_clusters.XData = cluster_centers(:,1);
        p_clusters.YData = cluster_centers(:,2);
        p_clusters.ZData = cluster_centers(:,3);
    else
        p_clusters.XData = NaN;
        p_clusters.YData = NaN;
        p_clusters.ZData = NaN;
    end

    p_unlabeled.XData = D_unlabeled(k,1:3:C);
    p_unlabeled.YData = D_unlabeled(k,2:3:C);
    p_unlabeled.ZData = D_unlabeled(k,3:3:C);
    
    p_labeled.XData = X(k,:);
    p_labeled.YData = Y(k,:);
    p_labeled.ZData = Z(k,:);
    
    %p_labeled.XData = augmented_frame(:,1);
    %p_labeled.YData = augmented_frame(:,2);
    %p_labeled.ZData = augmented_frame(:,3);
    
    drawnow 
    pause(0.1)
    %k
end

function cluster_centers = process_clusters(clusters)
    num_clusters = sum(~cellfun(@isempty,clusters),2);
    cluster_centers = zeros(num_clusters,3);
    for c = 1:num_clusters
        cluster_centers(c,:) = mean(clusters{1,c},1);
    end
end

function augmented_frame = assign_clusters(cluster_centers, curr_frame, prev_frame, eps)
    augmented_frame = curr_frame;
    K = size(cluster_centers,1);
    for k = 1:K
        [isBird, idx] = isInEpsVicinity(cluster_centers(k,:), prev_frame, eps);
        if isBird && isnan(curr_frame(idx,1))
           augmented_frame(idx,:) = cluster_centers(k,:);
        end
    end
end
% TODO manchmal ist der (ausversehen) nächste punkt besetzt, dann nimm
% zweit nächsten Punkt
function [bool, idx] = isInEpsVicinity(point, point_set, eps)
    d = sqrt(sum((point_set - point).^2, 2));
    [min_val, idx] = min(d);
    bool =  min_val < eps;
    if ~bool
        min_val
    end
end