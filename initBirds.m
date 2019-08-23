%% load data
load('rawDetections.mat')


%% Get patterns
allPatterns = read_patterns('tracking/datasets/framework');
patterns = zeros(10,4,3);
idx = 1;
for i=1:length(allPatterns)
    patterns(idx,:,:) = allPatterns(i).pattern;
    idx = idx + 1;
end
T = 200;
estimatedPositions = NaN * zeros(10,T,3);
estimatedQuats = NaN * zeros(10,T,4);

for t=1:T
%% initialize tracks

detections = squeeze(formattedData(1011+130+t,:,:));
detections = reshape(detections(~isnan(detections)),[],3);

epsilon = 50;
clustersRaw = clusterUnassignedDetections(detections, epsilon);
nClusters = 1;
clusters = {};
for i=1:length(clustersRaw)
    if size(clustersRaw{i},1) == 4
        if size(clustersRaw{i},1) > 4
            %TODO remove markers
        end
        clusters{nClusters} = clustersRaw{i};
        nClusters = nClusters + 1;
    end
end
nClusters = nClusters - 1;

if nClusters < 1
    fprintf('hi')
end
unassignedPatterns = ones(10,1);
costMatrix = zeros(sum(unassignedPatterns), length(clusters));
rotMatsMatrix = zeros(sum(unassignedPatterns), length(clusters), 3,3);
translationsMatrix = zeros(sum(unassignedPatterns), length(clusters), 3);
unassignedPatternsIdx = find(unassignedPatterns);
for i = 1:sum(unassignedPatterns)
    for j = 1:length(clusters)
        pattern = squeeze(patterns(unassignedPatternsIdx(i),:,:));
        p = match_patterns(pattern, clusters{j}, 'noKnowledge');
        assignment = zeros(4,1);
        assignment(p) = 1:length(p);
        pattern = pattern(assignment,:);
        dets = clusters{j};
        %size(dets,1)
        % TODO augment missing detection
        [R, translation, MSE] = umeyama(pattern', dets');
        costMatrix(i,j) = MSE;
        rotMatsMatrix(i,j,:,:) = R;
        translationsMatrix(i,j,:) = translation;
    end
end

costOfNonAssignment = 10; %TODO find something reasonable, altough could be very high
[patternToClusterAssignment, stillUnassignedPatterns, ~] = ...
    assignDetectionsToTracks(costMatrix, costOfNonAssignment);

for i=1:length(patternToClusterAssignment)
   patternIdx = patternToClusterAssignment(i,1);
   clusterIdx = patternToClusterAssignment(i,2);
   estimatedPositions(patternIdx,t, :) = translationsMatrix(patternIdx, clusterIdx,:);
   estimatedQuats(patternIdx, t,  :) = rotm2quat(squeeze(rotMatsMatrix(patternIdx, clusterIdx, :, :)));
end
end
%%
vizRes(formattedData(1011+130:1011+130+T,:,:), patterns, estimatedPositions, estimatedQuats, 0)

% TODO recover rotation from assignemnt with umeyama


%% do tracking