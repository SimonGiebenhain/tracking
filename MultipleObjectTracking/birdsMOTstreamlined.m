%% load data and patterns
dataFilename = 'datasets/session5/Starling_Trials_07-12-2019_09-30-00_Trajectories_100.csv';
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

stdHyperParams.costOfNonAsDtTA = 70;
stdHyperParams.certaintyFactor = 1;
stdHyperParams.useAssignmentLength = 1;
stdHyperParams.whenFPFilter = 70;
stdHyperParams.thresholdFPFilter = 50;
stdHyperParams.costOfNonAsDtMA = 10;
stdHyperParams.eucDistWeight = 1/3;
stdHyperParams.posNoise = 30;
stdHyperParams.motNoise = 30;
stdHyperParams.accNoise = 60;
stdHyperParams.quatNoise = 0.5;
stdHyperParams.quatMotionNoise = 0.05;
stdHyperParams.measurementNoise = 100;
stdHyperParams.certaintyScale = 3;
quatMotionType = 'brownian';

fprintf('Starting to track!\n')

%profile on
[estimatedPositions, estimatedQuats] = ownMOT(formattedData(:,:,:), patterns, patternNames ,0 , -1, 11, 0, -1, -1, quatMotionType, stdHyperParams);
%profile viewer

% Save the results

save([filePrefix, '_RESULTS.mat'], 'estimatedPositions', 'estimatedQuats')

