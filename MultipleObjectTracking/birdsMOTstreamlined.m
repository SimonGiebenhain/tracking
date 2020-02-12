%% load data and patterns
% Also add folder with patterns to path of matlab!
dataFilename = 'datasets/session1/all.csv'; %'datasets/session8/Starling_Trials_10-12-2019_16-00-00_Trajectories_100.csv'; % 'datasets/session1/all.csv';
patternDirectoryName = 'datasets/session1';
filePrefix = strsplit(dataFilename, '.');
filePrefix = filePrefix{1};
if isfile([filePrefix, '.mat'])
    load([filePrefix, '.mat']);
else
    % Also add folder with patterns to path of matlab!
    [formattedData, patternsPlusNames] = readVICONcsv(dataFilename, patternDirectoryName);
end
patterns = zeros(10,4,3);
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

stdHyperParams.costOfNonAsDtTA = 55;
stdHyperParams.certaintyFactor = 1;
stdHyperParams.useAssignmentLength = 1;
stdHyperParams.minAssignmentThreshold = 55;
stdHyperParams.costOfNonAsDtMA = 5;
stdHyperParams.eucDistWeight = 1/10;
stdHyperParams.posNoise = 10;
stdHyperParams.motNoise = 20;
stdHyperParams.accNoise = 20;
stdHyperParams.quatNoise = 0.05;
stdHyperParams.quatMotionNoise = 1;
stdHyperParams.measurementNoise = 65;
stdHyperParams.certaintyScale = 3;
quatMotionType = 'brownian';

fprintf('Starting to track!\n')

%profile on
beginningFrame = 1;%2500+7000;
[estimatedPositions, estimatedQuats] = ownMOT(formattedData(beginningFrame:end,:,:), patterns, patternNames ,0 , -1, size(patterns, 1), 0, -1, -1, quatMotionType, stdHyperParams);

%[estimatedPositions, estimatedQuats] = ownMOT(formattedData(1000:end,:,:), patterns, patternNames ,0 , -1, size(patterns, 1), 0, -1, -1, quatMotionType, stdHyperParams);

%profile viewer

% Save the results in .mat file format
%save([filePrefix, '_RESULTS.mat'], 'estimatedPositions', 'estimatedQuats')
%%
% Save file as .csv file, in VICON-style format
resultsFilename = [filePrefix '_RESULTS.csv'];
exportToCSV(resultsFilename, estimatedPositions, estimatedQuats, beginningFrame, patternNames, 1, 0);

