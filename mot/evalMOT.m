%% load data
%load('rawDetections.mat')
%load('D_labeled.mat')
load('modernMethods/data/matlab/generated_data.mat')
size(D)
N = size(D,1);
D = permute(D, [2, 1, 3, 4]);
formattedData = reshape(D, size(D,1), [], 3);


%% prepare data
initialStates = zeros( N, 3+3+4+4);
for i = 1:size(initialStates,1)
   initialStates(i,1:3) = pos(i, 1, :);
   initialStates(i,4:6) = zeros(1,3);
   initialStates(i,7:10) = quat(i, 1, :);
   initialStates(i, 11:14) = zeros(1,4);
end

%% Get patterns
allPatterns = read_patterns('tracking/datasets/framework');
patterns = zeros(10,4,3);
patternNames = {};
for i=1:length(allPatterns)
    patterns(i,:,:) = allPatterns(i).pattern;
    patternNames{i} = allPatterns(i).name;
end
    
    
%% test MOT
quatMotionType = 'brownian';
[estimatedPositions, estimatedQuats] = ownMOT(formattedData, patterns, patternNames ,initialStates, N, pos, quat, quatMotionType);

%TODO: save results

%% Evaluate tracking performance 
performanceVisualization(estimatedPositions, pos, estimatedQuats, quat, patterns);
%%
t0=1;
vizRes(formattedData(t0:end,:,:), patterns, estimatedPositions(:,t0:end,:), estimatedQuats(:,t0:end,:), 1, pos(:, t0:end, :), quat(:, t0:end, :))
