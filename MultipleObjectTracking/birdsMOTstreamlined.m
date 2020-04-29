%% load data and patterns
% Also add folder with patterns to path of matlab!
dataFilename = 'datasets/session8/Starling_Trials_10-12-2019_16-00-00_Trajectories_100.csv'; % 'datasets/session1/all.csv';
patternDirectoryName = 'datasets/session8';
filePrefix = strsplit(dataFilename, '.');
filePrefix = filePrefix{1};
if isfile([filePrefix, '.mat'])
    load([filePrefix, '.mat']);
else
    % Also add folder with patterns to path of matlab!
    [formattedData, patternsPlusNames] = readVICONcsv(dataFilename, patternDirectoryName);
end
patternsPlusNames = read_patterns(patternDirectoryName);
patterns = zeros(length(patternsPlusNames),4,3);
patternNames = {};
for i=1:length(patternsPlusNames)
    patterns(i,:,:) = patternsPlusNames(i).pattern;
    patternNames{i} = patternsPlusNames(i).name;
end
fprintf('Loaded data successfully!\n')


% idx = ones(11,1);
% idx(7) = 0;
% idx(11) = 0;
% idx = repmat(idx, 1, 4, 3);
% patterns = patterns(idx==1);
% patterns = reshape(patterns, 9, 4, 3);
% patternNames(11) = [];
% patternNames(7) = [];

%patterns(7,:,1) = 1000;
%patterns(11, :, 1) = 1000;


    
    
%% test MOT
stdHyperParams.doFPFiltering = 1;
stdHyperParams.adaptiveNoise = 1;
stdHyperParams.lambda = 0;
stdHyperParams.simplePatternMatching = 0;

stdHyperParams.costOfNonAsDtTA = 50; %session1: 85
stdHyperParams.certaintyFactor = 1;
stdHyperParams.useAssignmentLength = 1;
stdHyperParams.minAssignmentThreshold = 30; %30;
stdHyperParams.costOfNonAsDtMA = 10;
stdHyperParams.eucDistWeight = 1/3;%1/3;
stdHyperParams.posNoise = 10;
stdHyperParams.motNoise = 10;
stdHyperParams.accNoise = 5;
stdHyperParams.quatNoise = 0.052;
stdHyperParams.quatMotionNoise = 1;
stdHyperParams.measurementNoise = 50;
stdHyperParams.certaintyScale = 6.5;

stdHyperParams.modelType = 'LieGroup';

quatMotionType = 'brownian';

fprintf('Starting to track!\n')

%profile on
beginningFrame = 4000;%7800+ blau macht sehr komische sachen;5300 %+ 1000 jittery;%%2000+4000;
endFrame = size(formattedData,1);
stdHyperParams.visualizeTracking = 1;
[estimatedPositions, estimatedQuats, positionVariance, rotationVariance] = ownMOT(formattedData(beginningFrame:endFrame,:,:), patterns, patternNames ,0 , -1, size(patterns, 1), 0, -1, -1, quatMotionType, stdHyperParams);
%[estimatedPositions, estimatedQuats] = ownMOT(formattedData(1000:end,:,:), patterns, patternNames ,0 , -1, size(patterns, 1), 0, -1, -1, quatMotionType, stdHyperParams);


%%
colors = distinguishable_colors(10);
figure; hold on;
for i=1:10
    plot(smoothdata(positionVariance(i,:), 'movmedian', 30), 'color', colors(i,:))
end

hold off;
%%

%profile viewer

% Save the results in .mat file format
%save([filePrefix, '_RESULTS.mat'], 'estimatedPositions', 'estimatedQuats')
%%
% Save file as .csv file, in VICON-style format
resultsFilename = [filePrefix '_RESULTS.csv'];
exportToCSV(resultsFilename, estimatedPositions, estimatedQuats, beginningFrame, patternNames, 1, 0);

%%
vizParams.vizSpeed = 10;
vizParams.keepOldTrajectory = 0;
vizParams.vizHistoryLength = 500;
vizParams.startFrame = 7000;
vizParams.endFrame = -1;
vizRes(formattedData(beginningFrame:endFrame,:,:), patterns, estimatedPositions, estimatedQuats, vizParams, 0)


