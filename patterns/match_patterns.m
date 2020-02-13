function [assignment, lostDets, FPs, certainty, method] = match_patterns(whole_pattern, detected_pattern, method, rotMat, hyperParams)
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
lostDets = -1;
FPs = -1;
dim = size(whole_pattern,2);
if strcmp(method, 'correct')
    rotatedPattern = (rotMat*whole_pattern')';
    
    FPs = zeros(size(detected_pattern,1),1);
    
    % the rotated pattern also is the expected location of the markers
    % filter some False Positives
    
    if size(detected_pattern, 1) > 1 && hyperParams.doFPFiltering
        dists = pdist2(detected_pattern, rotatedPattern);
        if min(dists, [], 'all') < hyperParams.whenFPFilter
            internalD = squareform(pdist(detected_pattern));
            internalD(internalD == 0) = 10000;
            FPs = min(internalD, [],  2) > hyperParams.thresholdFPFilter;
            detected_pattern = detected_pattern(~FPs,:);
        end
    end
    
    if size(detected_pattern, 1) > 2
        method = 'new';
    else
        method = 'ML';
    end
    
    if hyperParams.simplePatternMatching == 1
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
        rotatedPattern = (rotMat*whole_pattern')';
        cost_matrix = pdist2(detected_pattern, rotatedPattern);
        [assignment, c] = munkers(cost_matrix);
        certainty = c;
    case 'new'
        sqDistPattern = squareform(pdist(whole_pattern));
        %anglesPattern = getInternalAngles(whole_pattern);
        
        %lambda = hyperParams.lambda;
        nMarkers = size(whole_pattern,1);
        allPerms = perms(1:nMarkers+1);
        cost = zeros(size(allPerms, 1),1);
        
        if size(detected_pattern,1) == 3
            det_pat = [detected_pattern; NaN*ones(1,3); NaN*ones(1,3)];
        else
            det_pat = [detected_pattern; NaN*ones(1,3)];
        end
        sqDistDetections = squareform(pdist(det_pat));
        
        for iii=1:size(allPerms,1)
            p = allPerms(iii,:);
            p = p(1:nMarkers);
            dets = det_pat(p,:);
            cost(iii) = sqrt(sum((sqDistDetections(p,p) - sqDistPattern).^2, 'all', 'omitnan'));
            
            cost(iii) = cost(iii) + lambda * sqrt(sum(angleDiff(~isnan(angleDiff))));
            
            
            eucDist = (dets - rotatedPattern).^2;
            eucDist = reshape( eucDist(~isnan(eucDist)), [], 3);
            cost(iii) = cost(iii) + hyperParams.eucDistWeight*sum(sqrt(sum(eucDist,2)));
            if sum(any(isnan(dets),2)) == 1 || sum(any(isnan(dets),2)) == 2
                cost(iii) = (cost(iii) + sum(any(isnan(dets),2)) * hyperParams.costOfNonAsDtMA);%/sum(all(~isnan(dets),2));
            elseif sum(any(isnan(dets),2)) == 0
                cost(iii) = cost(iii);%/sum(all(~isnan(dets),2));
            else
                fprintf('WTF')
            end
            
        end
        [c, minIdx] = min(cost);
        assignment = allPerms(minIdx,:);
        certainty = c;
    case 'final'
        if hyperParams.simplePatternMatching == 1
            rotatedPattern = (rotMat*whole_pattern')';
            cost_matrix = pdist2(detected_pattern, rotatedPattern);
            [assignment, c] = munkers(cost_matrix);
            certainty = c;
            method = 'ML';
        else
            rotatedPattern = (rotMat*whole_pattern')';
            sqDistPattern = squareform(pdist(whole_pattern));
            
            nDets = size(detected_pattern,1);
            nMarkers = size(whole_pattern,1);
            allPerms = perms(1:nMarkers);
            numPossibleDrops = 0;
            for indi=0:nDets-1
                numPossibleDrops = numPossibleDrops + nchoosek(nDets, indi);
            end
            cost = 1000000*ones(size(allPerms,1)*numPossibleDrops,1);
            Ps = cell(size(allPerms,1)*numPossibleDrops,1);
            Ds = cell(size(allPerms,1)*numPossibleDrops,1);
            
            
            sqDistDetections = squareform(pdist(detected_pattern));
            
            choiceCount = 1;
            for i1=1:size(allPerms,1)
                p = allPerms(i1,:);
                for j1=0:nDets-1
                    drops = nchoosek(1:nDets, j1);
                    for i2=1:size(drops, 1)
                        drop = drops(i2, :);
                        dets = detected_pattern;
                        dets(drop, :) = [];
                        nD = size(dets,1);
                        p = p(1:nD);
                        Ps{choiceCount} = p;
                        Ds{choiceCount} = drop;
                        
                        distsDet = sqDistDetections;
                        distsDet(drop, :) = [];
                        distsDet(:, drop) = [];
                        distsPat = sqDistPattern(p,p);
                        
                        cost(choiceCount) = mean(sqrt(sum((distsDet - distsPat).^2, 2)));
                        
                        eucDist = mean(sqrt(sum((dets - rotatedPattern(p, :)).^2, 2)));
                        
                        cost(choiceCount) = cost(choiceCount) + hyperParams.eucDistWeight*eucDist;
                        cost(choiceCount) = cost(choiceCount) + hyperParams.costOfNonAsDtMA * j1;
                        choiceCount = choiceCount + 1;
                        %TODO: maybe add penaltiy for dropping all except one
                        %detection
                    end
                end
            end
            [c, minIdx] = min(cost);
            assignment = Ps{minIdx};
            lostDets = Ds{minIdx};
            %assignment = invs{minIdx};
            certainty = c;
        end
    case 'final2'
        if hyperParams.simplePatternMatching == 1
            rotatedPattern = (rotMat*whole_pattern')';
            cost_matrix = pdist2(detected_pattern, rotatedPattern);
            [assignment, c] = munkers(cost_matrix);
            certainty = c;
            method = 'ML';
        else
            rotatedPattern = (rotMat*whole_pattern')';
            sqDistMarkers = squareform(pdist(whole_pattern));
            sqDistDetections = squareform(pdist(detected_pattern));
            
            nDets = size(detected_pattern,1);
            nMarkers = size(whole_pattern,1);
            % number of assignments that are checked
            numAssignments = 0;
            for indi =max(1,nDets-2):nDets
                numAssignments = numAssignments + nchoosek(nDets, indi)*nchoosek(nMarkers,indi)*factorial(indi);
            end
            
            cost = 1000000*ones(numAssignments,1);
            checkedPerms = cell(numAssignments,1);
            chosenMarkers = cell(numAssignments,1);
            chosenDets = cell(numAssignments,1);
            choiceCount = 1;
            %LOOPSLOOPSLOOPS
            for numPoints=max(1,nDets-2):nDets
                allPerms = perms(1:numPoints);
                allMarkerSelections = nchoosek(1:nMarkers, numPoints);
                allDetSelections = nchoosek(1:nDets, numPoints);
                for iD=1:size(allDetSelections,1)
                    detSelect = allDetSelections(iD,:);
                    dets = detected_pattern(detSelect,:);
                    sqDistD = sqDistDetections(detSelect,detSelect);
                    
                    for iM=1:size(allMarkerSelections,1)
                        mSelect = allMarkerSelections(iM,:);
                        rotatedP = rotatedPattern(mSelect,:);
                        sqDistM = sqDistMarkers(mSelect,mSelect);
                        
                        for iP=1:length(allPerms)
                            p = allPerms(iP,:);
                            checkedPerms{choiceCount} = p;
                            chosenMarkers{choiceCount} = mSelect;
                            chosenDets{choiceCount} = detSelect;
                            
                            cost(choiceCount) = mean(sqrt(sum((sqDistD - sqDistM(p,p)).^2, 2)));
                            
                            eucDist = mean(sqrt(sum((dets - rotatedP(p, :)).^2, 2)));
                            
                            cost(choiceCount) = cost(choiceCount) + hyperParams.eucDistWeight*eucDist;
                            cost(choiceCount) = cost(choiceCount) + hyperParams.costOfNonAsDtMA * (nDets-numPoints);
                            choiceCount = choiceCount + 1;
                        end
                    end
                end
            end
            
            [c, minIdx] = min(cost);
            %assignment = Ps{minIdx};
            chosenMs = chosenMarkers{minIdx};
            assignment = chosenMs(checkedPerms{minIdx});
            %TODO I have to incorporate
            
            %lostDets = Ds{minIdx};
            lostDets = setdiff(1:nDets, chosenDets{minIdx});
            certainty = c;
        end
    case 'final3'
        if hyperParams.simplePatternMatching == 1
            rotatedPattern = (rotMat*whole_pattern')';
            cost_matrix = pdist2(detected_pattern, rotatedPattern);
            [assignment, c] = munkers(cost_matrix);
            certainty = c;
            method = 'ML';
        else
            rotatedPattern = (rotMat*whole_pattern')';
            sqDistMarkers = squareform(pdist(whole_pattern));
            sqDistDetections = squareform(pdist(detected_pattern));
            
            nDets = size(detected_pattern,1);
            nMarkers = size(whole_pattern,1);
            % number of assignments that are checked
            numAssignments = 0;
            for indi =max(2,nDets-2):nDets
                numAssignments = numAssignments + nchoosek(nDets, indi)*nchoosek(nMarkers,indi)*factorial(indi);
            end
            
            cost = 1000000*ones(numAssignments,1);
            checkedPerms = cell(numAssignments,1);
            chosenMarkers = cell(numAssignments,1);
            chosenDets = cell(numAssignments,1);
            choiceCount = 1;
            %LOOPSLOOPSLOOPS
            for numPoints=max(2,nDets-2):nDets
                allPerms = perms(1:numPoints);
                allMarkerSelections = nchoosek(1:nMarkers, numPoints);
                allDetSelections = nchoosek(1:nDets, numPoints);
                for iD=1:size(allDetSelections,1)
                    detSelect = allDetSelections(iD,:);
                    dets = detected_pattern(detSelect,:);
                    sqDistD = sqDistDetections(detSelect,detSelect);
                    
                    for iM=1:size(allMarkerSelections,1)
                        mSelect = allMarkerSelections(iM,:);
                        rotatedP = rotatedPattern(mSelect,:);
                        sqDistM = sqDistMarkers(mSelect,mSelect);
                        
                        for iP=1:length(allPerms)
                            p = allPerms(iP,:);
                            checkedPerms{choiceCount} = p;
                            chosenMarkers{choiceCount} = mSelect;
                            chosenDets{choiceCount} = detSelect;
                            
                            cost(choiceCount) = mean(vecnorm(sqDistD - sqDistM(p,p),2,2));
                            
                            eucDist = mean(vecnorm(dets - rotatedP(p, :),2,2));
                            
                            cost(choiceCount) = cost(choiceCount) + hyperParams.eucDistWeight*eucDist;
                            cost(choiceCount) = cost(choiceCount) + hyperParams.costOfNonAsDtMA * (nDets-numPoints);
                            choiceCount = choiceCount + 1;
                        end
                    end
                end
            end
            
            [c, minIdx] = min(cost);
            %assignment = Ps{minIdx};
            chosenMs = chosenMarkers{minIdx};
            assignment = chosenMs(checkedPerms{minIdx});
            %TODO I have to incorporate
            
            %lostDets = Ds{minIdx};
            lostDets = setdiff(1:nDets, chosenDets{minIdx});
            certainty = c;
        end
    case 'final4'
        if hyperParams.simplePatternMatching == 1
            rotatedPattern = (rotMat*whole_pattern')';
            cost_matrix = pdist2(detected_pattern, rotatedPattern);
            [assignment, c] = munkers(cost_matrix);
            certainty = c;
            method = 'ML';
        else
            
            minNumDetsToConsider = 1;
            maxNumFPsToConsider = 3;
            
            rotatedPattern = (rotMat*whole_pattern')';
            sqDistMarkers = squareform(pdist(whole_pattern));
            sqDistDetections = squareform(pdist(detected_pattern));
            
            nDets = size(detected_pattern,1);
            nMarkers = size(whole_pattern,1);
            % number of assignments that are checked
            numAssignments = 0;
            for indi =max(minNumDetsToConsider,nDets-maxNumFPsToConsider):nDets
                numAssignments = numAssignments + nchoosek(nDets, indi)*nchoosek(nMarkers,indi)*factorial(indi);
            end
            
            cost = 1000000*ones(numAssignments,1);
            checkedPerms = cell(numAssignments,1);
            chosenMarkers = cell(numAssignments,1);
            chosenDets = cell(numAssignments,1);
            choiceCount = 1;
            %LOOPSLOOPSLOOPS
            for numPoints=max(minNumDetsToConsider,nDets-maxNumFPsToConsider):nDets
                allPerms = perms(1:numPoints);
                allMarkerSelections = nchoosek(1:nMarkers, numPoints);
                allDetSelections = nchoosek(1:nDets, numPoints);
                chunkSize = nchoosek(nDets, numPoints)*nchoosek(nMarkers,numPoints)*factorial(numPoints);
                beginningOfChunk = choiceCount;
                endOfChunk = beginningOfChunk + chunkSize - 1;
                leftChunkEuc = zeros(chunkSize,numPoints,3);
                rightChunkEuc = zeros(chunkSize,numPoints,3);
                leftChunkInternal = zeros(chunkSize,numPoints,numPoints);
                rightChunkInternal = zeros(chunkSize,numPoints,numPoints);
                chunkIdx = 1;
                for iD=1:size(allDetSelections,1)
                    detSelect = allDetSelections(iD,:);
                    dets = detected_pattern(detSelect,:);
                    sqDistD = sqDistDetections(detSelect,detSelect);
                    for iM=1:size(allMarkerSelections,1)
                        mSelect = allMarkerSelections(iM,:);
                        rotatedP = rotatedPattern(mSelect,:);
                        sqDistM = sqDistMarkers(mSelect,mSelect);
                        
                        for iP=1:length(allPerms)
                            p = allPerms(iP,:);
                            checkedPerms{choiceCount} = p;
                            chosenMarkers{choiceCount} = mSelect;
                            chosenDets{choiceCount} = detSelect;
                            leftChunkInternal(chunkIdx, :, :) = sqDistD;
                            rightChunkInternal(chunkIdx, :, :) = sqDistM(p,p);
                            
                            leftChunkEuc(chunkIdx, :, :) = dets;
                            rightChunkEuc(chunkIdx, :, :) = rotatedP(p,:);
                            chunkIdx = chunkIdx + 1;
                            
                            %cost(choiceCount) = mean(vecnorm(sqDistD - sqDistM(p,p),2,2));
                            
                            %eucDist = mean(vecnorm(dets - rotatedP(p, :),2,2));
                            
                            %cost(choiceCount) = cost(choiceCount) + hyperParams.eucDistWeight*eucDist;
                            cost(choiceCount) = hyperParams.costOfNonAsDtMA * (nDets-numPoints);
                            choiceCount = choiceCount + 1;
                        end
                    end
                end
                internalDist = mean(vecnorm(leftChunkInternal - rightChunkInternal, 2, 3), 2);
                eucDist = mean(vecnorm(leftChunkEuc - rightChunkEuc, 2, 3), 2);
                eucWeight = hyperParams.eucDistWeight;
                if numPoints <= 2
                    eucWeight = eucWeight*1.5;
                elseif numPoints == 4
                    eucWeight = eucWeight/2;
                end
                cost(beginningOfChunk:endOfChunk) = cost(beginningOfChunk:endOfChunk) + internalDist + eucWeight*eucDist;
            end
            
            [c, minIdx] = min(cost);
            %assignment = Ps{minIdx};
            chosenMs = chosenMarkers{minIdx};
            assignment = chosenMs(checkedPerms{minIdx});
            %TODO I have to incorporate
            
            %lostDets = Ds{minIdx};
            lostDets = setdiff(1:nDets, chosenDets{minIdx});
            certainty = c;
        end
    case 'noKnowledge'
        costNonAssignment = 50;
        nMarkers = size(whole_pattern,1);
        allPerms = perms(1:nMarkers);
        cost = zeros(size(allPerms, 1),1);
        if size(detected_pattern,1) == 3
            det_pat = [detected_pattern; NaN*ones(1,3)];
        else
            det_pat = detected_pattern;
        end
        
        %anglesPattern = getInternalAngles(whole_pattern);
        sqDistPattern = squareform(pdist(whole_pattern));
        sqDistDetections = squareform(pdist(det_pat));
        
        
        for iii=1:size(allPerms,1)
            p = allPerms(iii,:);
            p = p(1:nMarkers);
            dets = det_pat(p,:);
            
            cost(iii) = sqrt(sum((sqDistDetections(p,p) - sqDistPattern).^2, 'all', 'omitnan'));
            %anglesDetections = getInternalAngles(dets);
            %angleDiff = (anglesDetections - anglesPattern).^2;
            %cost(iii) = cost(iii) + lambda * sqrt(sum(angleDiff(~isnan(angleDiff))));
            
            
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