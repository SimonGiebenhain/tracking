function assignment = match_patterns(whole_pattern, detected_pattern)
%MATCH_PATTERNS Match points within two sets of points
%   So far this method relies on a constant/known orientation of the point
%   pattern.
%   assignment(i) contains index of marker in whole_pattern assigned to the
%   ith detected marker.
%
%   The algorithm creates edges representing the offset from one point to
%   another. Then it compares these edges within the two sets of points and
%   finds correspondances where the outgoing edges of a point are similar
%   in both point sets.
%   The algorithm is very simple and only a first draft.

    edges1 = get_edges(whole_pattern);
    self_edges_idx = repmat(sum(abs(edges1),3) == 0,1,1,2);
    edges1(self_edges_idx) = Inf;
    edges2 = get_edges(detected_pattern);

    num_nodes_1 = size(edges1,1);
    num_nodes_2 = size(edges2,1);
    cost_matrix = zeros(num_nodes_2, num_nodes_1);
    
    for i = 1:num_nodes_2
        for j = 1:num_nodes_1
            if size(edges1,3) == 1
                e2 = edges2(i,:)';
                e1 = edges1(j,:)';
            else
                e2 = squeeze(edges2(i,:,:));
                e1 = squeeze(edges1(j,:,:));
            end
            c = 0;
            for k = 1:size(e2,1)
                if sum(e2(k,:)) == 0
                    continue;
                end
                diff = sqrt(sum((e1-e2(k,:)).^2,2));
                [min_val, min_idx] = min(diff);
                c = c + min_val;
                e1(min_idx,:) = Inf;
            end
            cost_matrix(i,j) = c;
        end
    end
    [assignment, bi_cost] = munkers(cost_matrix)
    
    function leaving_edges = get_edges(marker)
    dim = size(marker,2);
    num_markers = size(marker,1);
    leaving_edges = zeros(num_markers,num_markers,dim);

    for ii=1:num_markers
        for jj=1:num_markers
            if ii == jj
                continue;
            end
            leaving_edges(ii,jj,:) = marker(jj,:) - marker(ii,:);
        end
    end
end


end