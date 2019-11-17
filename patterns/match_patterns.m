function [assignment, FPs, certainty] = match_patterns(whole_pattern, detected_pattern, method, kalmanFilter)
%MATCH_PATTERNS Find corresponing points in two sets of points
%
%   assignment = MATCH_PATTERNS(whole_pattern, detected_pattern, 'ML', rotMat)
%
%   When using the 'ML' method the function expects two roughly centered
%   sets of points. Then whole_pattern is rotated by the rotation Matrix
%   Rot. The assignment is then based on the distance between the points
%   and solved with the Hungarian Algorithm.
%
%   The following method is outdated.
%
%   assignment = MATCH_PATTERNS(whole_pattern, detected_pattern, 'edges')
%
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

dim = size(whole_pattern,2);

if strcmp(method, 'correct')
    rotMat = Rot(kalmanFilter.x(2*dim+1:2*dim+4));
    rotatedPattern = (rotMat*whole_pattern')';

    FPs = zeros(size(detected_pattern,1),1);

    % the rotated pattern also is the expected location of the markers
    % filter some False Positives
    
    if size(detected_pattern, 1) > 1
       dists = pdist2(detected_pattern, rotatedPattern);
       if min(dists, [], 'all') < 20
          internalD = squareform(pdist(detected_pattern));
          internalD(internalD == 0) = 100;
          FPs = min(internalD, [],  2) > 50;
          detected_pattern = detected_pattern(~FPs,:);
       end
    end

    if size(detected_pattern, 1) > 2
        method = 'new';
    else
        method = 'ML';
    end
end

switch method
    case 'edges'
        edges1 = get_edges(whole_pattern);
        dim = size(edges1,3);
        self_edges_idx = repmat(sum(abs(edges1),3) == 0,1,1,dim);
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
        [assignment, bi_cost] = munkers(cost_matrix);
    case 'ML'
        
        cost_matrix = pdist2(detected_pattern, rotatedPattern);
        [assignment, c] = munkers(cost_matrix);
        certainty = c;
    case 'new'
        %TODO make this cost of NonAssignemnt smaller!!!
        costNonAssignment = 10;
                
        sqDistPattern = squareform(pdist(whole_pattern));
        anglesPattern = getInternalAngles(whole_pattern);

        lambda = 20;
        
        nMarkers = size(whole_pattern,1);
        allPerms = perms(1:nMarkers+1);
        cost = zeros(size(allPerms, 1),1);
        
        if size(detected_pattern,1) == 3
               det_pat = [detected_pattern; NaN*ones(1,3); NaN*ones(1,3)];
           else
               det_pat = [detected_pattern; NaN*ones(1,3)];
        end
        sqDistDetections = squareform(pdist(det_pat));
        %anglesDetections = getInternalAngles(det_pat);

           
        for iii=1:size(allPerms,1)
           p = allPerms(iii,:);
           p = p(1:nMarkers);
           dets = det_pat(p,:);
           cost(iii) = sqrt(sum((sqDistDetections(p,p) - sqDistPattern).^2, 'all', 'omitnan'));
           
           %TODO include or not?
           anglesDetections = getInternalAngles(dets);
           angleDiff = (anglesDetections - anglesPattern).^2;
           cost(iii) = cost(iii) + lambda * sqrt(sum(angleDiff(~isnan(angleDiff))));
           
           
           %for in = 1:4
           %    if all(~isnan(dets(in,:)))
           %        x = kalmanFilter.x;
           %        F = HLin(in:4:end,:);
           %        covMat = F * kalmanFilter.P * F';
           %        negLogL = -reallog(mvnpdf(dets(in,:), (F*x)', round(covMat,4)))';
           %        %if negLogL < Inf
           %        %    negLogL
           %        %end
           %        cost(iii) = cost(iii) + sum(negLogL);
           %    end
           %end
           
           eucDist = (dets - rotatedPattern).^2;
           eucDist = reshape( eucDist(~isnan(eucDist)), [], 3);
           cost(iii) = cost(iii) + 1/10*sum(sqrt(sum(eucDist,2)));
           if sum(any(isnan(dets),2)) == 1 || sum(any(isnan(dets),2)) == 2
                cost(iii) = (cost(iii) + sum(any(isnan(dets),2))* costNonAssignment)/sum(all(~isnan(dets),2));
           elseif sum(any(isnan(dets),2)) == 0
                cost(iii) = cost(iii)/sum(all(~isnan(dets),2));
           else
               fprintf('WTF')
           end
           
        end
        [c, minIdx] = min(cost);
        assignment = allPerms(minIdx,:);
        certainty = c;
        %assignment = assignment(1:nMarkers);
        %assignment = assignment(assignment ~= 5);
    case 'noKnowledge'
        costNonAssignment = 50;
                   
        lambda = 20;
        nMarkers = size(whole_pattern,1);
        allPerms = perms(1:nMarkers);
        cost = zeros(size(allPerms, 1),1);
        for iii=1:size(allPerms,1)
           p = allPerms(iii,:);
           %dets = NaN * zeros(nMarkers, 3);
           if size(detected_pattern,1) == 3
               p = p(1:nMarkers);
               det_pat = [detected_pattern; NaN*ones(1,3)];
               dets = det_pat(p,:);
           else
               p = p(1:nMarkers);
               det_pat = detected_pattern;
               dets = det_pat(p,:);
           end
           distDetections = pdist(dets);
           distPattern = pdist(whole_pattern);
           diff = abs(distDetections - distPattern);
           cost(iii) = sum(diff(~isnan(diff)), 'all');
           anglesDetections = getInternalAngles(dets);
           anglesPattern = getInternalAngles(whole_pattern);
           angleDiff = abs(anglesDetections - anglesPattern);
           cost(iii) = cost(iii) + lambda * sum(angleDiff(~isnan(angleDiff)), 'all');

           if sum(any(isnan(dets),2)) == 1 || sum(any(isnan(dets),2)) == 2
                cost(iii) = (cost(iii) + sum(any(isnan(dets),2))* costNonAssignment)/sum(all(~isnan(dets),2));
           elseif sum(any(isnan(dets),2)) == 0
                cost(iii) = cost(iii)/sum(all(~isnan(dets),2));
           else
               fprintf('WTF')
           end
           
        end
        [c, minIdx] = min(cost);
        assignment = allPerms(minIdx,:);
end

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

function angles = getInternalAngles(points)
    N = size(points,1);
    allTriples = nchoosek(1:N, 3); % TODO since only called with 3 or 4 this could be precomputed
    nTriples = size(allTriples, 1);
    angles = zeros(3*nTriples);
    for in = 0:nTriples-1
        p1 = points(allTriples(in+1,1), :);
        p2 = points(allTriples(in+1,2), :);
        p3 = points(allTriples(in+1,3), :);
        
        v1 = p2 - p1;
        v2 = p3 - p1;
        angles(in*3+2) = atan2(norm(cross(v1,v2)),dot(v1,v2));
        
        v1 = p1 - p2;
        v2 = p3 - p2;
        angles(in*3+1) = atan2(norm(cross(v1,v2)),dot(v1,v2));
        
        v1 = p1 - p3;
        v2 = p2 - p3;
        angles(in*3+3) = atan2(norm(cross(v1,v2)),dot(v1,v2));
    end
end

%function angle = getAngle(v1, v2)
%    vn = cross(v1,v2);
%    normVn = sqrt(sum(vn.^2));
%    angle = atan2(normVn, dot(v1, v2));
%end

end