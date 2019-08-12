allPatterns = read_patterns('tracking/datasets/framework');
numWrong = 0;
for iter=1:10000
patternIdx = randi(10);
pattern = allPatterns(patternIdx).pattern;
perm = randperm(size(pattern,1));

rotX = randi(360);
rotY = randi(360);
rotZ = randi(360);
rotMatX = rotx(180);
rotMatY = roty(180);
rotMatZ = rotz(180);

patternPrime = pattern(perm,:) + 1*randn(size(pattern));
patternPrime = (rotMatZ * rotMatY * rotMatX * patternPrime')';

ass = match(pattern(:,:), patternPrime(:,:));

if any(ass ~= perm)
    figure;
    scatter3(pattern(1:3,1), pattern(1:3,2), pattern(1:3,3)); hold on;
    scatter3(pattern(4,1), pattern(4,2), pattern(4,3), 'MarkerEdgeColor', 'black');
    scatter3(patternPrime(:,1), patternPrime(:,2), patternPrime(:,3), 'MarkerEdgeColor', 'red')
    %fprintf('WRONF')
    %break
    numWrong = numWrong + 1
end
end
numWrong

function [assignment] = match(pattern, detections)
    lambda = 30;
    nMarkers = size(pattern,1);
    allPerms = perms(1:nMarkers);
    cost = zeros(size(allPerms, 1),1);
    for i=1:size(allPerms,1)
       p = allPerms(i,:);
       dets = NaN * zeros(nMarkers, 3);
       if size(detections,1) == 3
            dets(p,:) = [detections; NaN*zeros(1,3)];
       else
           dets(p,:) = detections;
       end
       distDetections = pdist(dets);
       distPattern = pdist(pattern);
       diff = abs(distDetections - distPattern);
       cost(i) = sum(diff(~isnan(diff)), 'all');
       anglesDetections = getInternalAngles(dets);
       anglesPattern = getInternalAngles(pattern);
       angleDiff = abs(anglesDetections - anglesPattern);
       cost(i) = cost(i) + lambda * sum(angleDiff(~isnan(angleDiff)), 'all');
    end
    [c, minIdx] = min(cost);
    assignment = allPerms(minIdx,:);
end


function angles = getInternalAngles(points)
    N = size(points, 1);
    angles = zeros(N, nchoosek(N,2));
    allPairs = nchoosek(1:N,2);
    for i = 1:size(angles, 1)
        for j = 1:size(angles,2)
            p1 = points(allPairs(j,1),:);
            p2 = points(allPairs(j,2),:);
            v1 = p1 - points(i,:);
            v2 = p2 - points(i,:);
            angles(i,j) = getAngle(v1, v2);
        end
    end
end

function angle = getAngle(v1, v2)
    vn = cross(v1,v2);
    normVn = sqrt(sum(vn.^2));
    angle = atan2(normVn, dot(v1, v2));
end