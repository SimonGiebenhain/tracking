allPatterns = read_patterns('tracking/datasets/framework');

t = 1;
bird = 7;
% get the detections for frame 1
dets = squeeze(formattedData(t,:,:));

% get vicons estimate for bird number 9 in frame 1
pattern = allPatterns(bird).pattern;
pos = squeeze(viconTrajectories(bird,t,:));
quat = squeeze(viconOrientation(bird,t,:));
%quat = [quat(4); quat(1:3)];
rotatedPattern = (Rot(quat) * pattern')';
markerPositions = pos' + rotatedPattern;


isNear = sqrt(sum((dets - pos').^2, 2)) < 100;
dets = dets(isNear,:);


%markerPositions = markerPositions([1;4;2;3],:)

figure;
plot3(markerPositions(:,1), markerPositions(:,2), markerPositions(:,3), 'o', 'MarkerSize', 10);
hold on;
plot3(dets(:,1), dets(:,2), dets(:,3), '*', 'MarkerSize', 10, 'MarkerEdgeColor', [0.5 0.5 0.5]);
%plot3(markerPositions(1,1), markerPositions(1,2), markerPositions(1,3), 's', 'MarkerSize', 20);
%plot3(dets(1,1), dets(1,2), dets(1,3), 's', 'MarkerSize', 20);
grid on;
axis equal
