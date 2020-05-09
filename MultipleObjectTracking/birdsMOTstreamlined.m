%% load data and patterns
% Also add folder with patterns to path of matlab!
dataFilename = 'datasets/session8/Starling_Trials_10-12-2019_16-00-00_Trajectories_100.csv'; % 'datasets/session1/all.csv'; %
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

% What was changed:
%   - certainty Scale : 6.5 -> 7.5
%   - minAssThresh: 30 -> 25
%   - new init: 0.5 -> 0.7
%   - FP-filtering: never throw away all dets
%   - abstand ghost init: 80 -> 90

% Idea:
%   - lower measurement error (maybe remove isFlying indicator, which
%   increases measurment error while flying)
%   - work on backwards pass and max acc. and vel.
%   - (try switching motion model for ghost birds; already decent without)

%TODO: hin und her wackeln irgendwie vermeiden!!
%TODO: wenn alle bis auf einen; leichter identifizieren -> test wie match
%patterns zwischeneinander abschließt
%TODO: certainty speichern: zB. bei 4 detections wie gut ist der fit ->
%   iD-switches könnten damit bemerkt werden
%TODO: wie kann man verhindern, dass von detections weg beschleunigt wird?


%8200 id-sw.
%8480 grün geht beim fliegen verloren, was da los??
stdHyperParams.costOfNonAsDtTA = 50;%85 %session8: 50, alt: flying birds get lower assignment cost
stdHyperParams.certaintyFactor = 1;
stdHyperParams.useAssignmentLength = 1;
stdHyperParams.minAssignmentThreshold = 30; %35%30;
stdHyperParams.ghostFPFilterDist = 65;
stdHyperParams.costOfNonAsDtMA = 10;
stdHyperParams.eucDistWeight = 1/4;%1/3;
stdHyperParams.posNoise = 50;%%110;%60;%50
stdHyperParams.motNoise = 20;%;1;%5;%10
stdHyperParams.accNoise = 10;%0.1;%1;%3
stdHyperParams.quatNoise = 0.2;
stdHyperParams.quatMotionNoise = 1;
stdHyperParams.measurementNoise = 70;%50
stdHyperParams.certaintyScale = 1;%6.5

stdHyperParams.minDistToBird = 95; %minimal distance for new ghost birds to other (ghost) birds that has to be free.
stdHyperParams.initThreshold = 1.1;%0.85;
stdHyperParams.initThreshold4 = 2.5;
stdHyperParams.patternSimilarityThreshold = 1.2;%1;

stdHyperParams.modelType = 'LieGroup';

quatMotionType = 'brownian';

fprintf('Starting to track!\n')

%profile on
beginningFrame = 4000;%4000;%7800+ blau macht sehr komische sachen;5300 %+ 1000 jittery;%%2000+4000;
endFrame = size(formattedData,1);
stdHyperParams.visualizeTracking = 1;
tic
[estimatedPositions, estimatedQuats, positionVariance, rotationVariance] = ownMOT(formattedData(beginningFrame:endFrame,:,:), patterns, patternNames ,0 , -1, size(patterns, 1), 0, -1, -1, quatMotionType, stdHyperParams);
%[estimatedPositions, estimatedQuats] = ownMOT(formattedData(1000:end,:,:), patterns, patternNames ,0 , -1, size(patterns, 1), 0, -1, -1, quatMotionType, stdHyperParams);
toc
%%

reverseIdx = sort(beginningFrame:endFrame, 'descend');
formattedDataRev = formattedData(reverseIdx, :, :);
tic
[estimatedPositionsRev, estimatedQuatsRev] = ownMOTbackward(formattedDataRev, patterns, patternNames, initialStates, 10, 0, estimated, forwardRot, quatMotionType, hyperParams)

[estimatedPositionsRev, estimatedQuatsRev, positionVarianceRev, rotationVarianceRev] = ownMOT(formattedDataRev, patterns, patternNames ,0 , -1, size(patterns, 1), 0, -1, -1, quatMotionType, stdHyperParams);
toc


%%
colors = distinguishable_colors(10);
figure; hold on;
for i=1:10
    plot(min(smoothdata(positionVariance(i,:), 'movmedian', 30), 1000), 'color', colors(i,:))
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
vizParams.startFrame = 1;
vizParams.endFrame = -1;

%reverseIdx = sort(1:size(estimatedPositionsRev, 2), 'descend');
vizRes(formattedData(beginningFrame:endFrame,:,:), patterns, estimatedPositions, estimatedQuats, vizParams, 0)
    %estimatedPositionsRev(:, reverseIdx,  :), estimatedQuatsRev(:, reverseIdx, :))


