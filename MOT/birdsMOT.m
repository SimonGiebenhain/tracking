%% load data
load('rawDetections.mat')
load('D_labeled.mat')

%% prepare data
offset = 130;
formattedData = formattedData(1011+offset-1:end, :,:);

initialData = D_labeled(offset,:);

initialStates = zeros( length(initialData)/7, 3+3+4+4 );
for i = 1:size(initialStates,1)
   initialStates(i,1:3) = initialData( 7*(i-1)+5:i*7 );
   initialStates(i,4:6) = zeros(1,3);
   initialStates(i,7:10) = initialData(7*(i-1)+1:(i-1)*7+4);
end

initialStates = initialStates( sum(isnan(initialStates),2) == 0, :);

%% Get patterns, bird 5 and 6 are missing in the beginning, don't track them for now
allPatterns = read_patterns('tracking/datasets/framework');
patterns = zeros(10,4,3);
idx = 1;
for i=1:length(allPatterns)
    patterns(idx,:,:) = allPatterns(i).pattern;
    idx = idx + 1;
end
    
    
%% test MOT
[estimatedPositions, estimatedQuats] = ownMOT(formattedData, patterns, initialStates, 10);

%% Evaluate tracking performance 
% Plot the estimation error of the positions and orientations
performanceVisualization(estimatedPositions, positions, estimatedQuats, quats);