%% load data and patterns
dataFilename = 'datasets/session5/Starling_Trials_07-12-2019_09-30-00_Trajectories_100.csv'; % 'datasets/session1/all.csv';
patternDirectoryName = 'datasets/session2';
filePrefix = strsplit(dataFilename, '.');
filePrefix = filePrefix{1};
if isfile([filePrefix, '.mat'])
    load([filePrefix, '.mat']);
else
    [formattedData, patternsPlusNames] = readVICONcsv(dataFilename, patternDirectoryName);
end
patterns = zeros(10,4,3);
patternNames = {};
for i=1:length(patternsPlusNames)
    patterns(i,:,:) = patternsPlusNames(i).pattern;
    patternNames{i} = patternsPlusNames(i).name;
end
fprintf('Loaded data successfully!\n')

    
    
%% test MOT
stdHyperParams.doFPFiltering = 1;
stdHyperParams.adaptiveNoise = 1;
stdHyperParams.lambda = 0;
stdHyperParams.simplePatternMatching = 0;

stdHyperParams.costOfNonAsDtTA = 100;
stdHyperParams.certaintyFactor = 1;
stdHyperParams.useAssignmentLength = 1;
stdHyperParams.minAssignmentThreshold = 65;
stdHyperParams.costOfNonAsDtMA = 10;
stdHyperParams.eucDistWeight = 1/5;
stdHyperParams.posNoise = 10;
stdHyperParams.motNoise = 20;
stdHyperParams.accNoise = 20;
stdHyperParams.quatNoise = 1;
stdHyperParams.quatMotionNoise = 1;
stdHyperParams.measurementNoise = 65;
stdHyperParams.certaintyScale = 3;
quatMotionType = 'brownian';

fprintf('Starting to track!\n')

profile on
[estimatedPositions, estimatedQuats] = ownMOT(formattedData(42000:end,:,:), patterns, patternNames ,0 , -1, size(patterns, 1), 0, -1, -1, quatMotionType, stdHyperParams);

%[estimatedPositions, estimatedQuats] = ownMOT(formattedData(1000:end,:,:), patterns, patternNames ,0 , -1, size(patterns, 1), 0, -1, -1, quatMotionType, stdHyperParams);

profile viewer

% Save the results

%save([filePrefix, '_RESULTS.mat'], 'estimatedPositions', 'estimatedQuats')

