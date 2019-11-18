%% Parameter settings to test

stdHyperParams.doFPFiltering = 1;
stdHyperParams.adaptiveNoise = 1;
stdHyperParams.lambda = 20;
stdHyperParams.simplePatternMatching = 0;

stdHyperParams.costOfNonAsDtTA = 70;
stdHyperParams.certaintyFactor = 3;
stdHyperParams.useAssignmentLength = 1;
stdHyperParams.whenFPFilter = 20;
stdHyperParams.thresholdFPFilter = 50;
stdHyperParams.costOfNonAsDtMA = 10;
stdHyperParams.eucDistWeight = 1/10;
stdHyperParams.posNoise = 20;
stdHyperParams.motNoise = 20;
stdHyperParams.quatNoise = 0.05;
stdHyperParams.quatMotionNoise = 0.05;
stdHyperParams.measurementNoise = 50;

testNames = {'none', 'all', 'noFPFiltering', 'noAdaptiveNoise', 'noAngles', 'simplePatternMatching'};
hyperParamsNone = stdHyperParams;
hyperParamsNone.doFPFiltering = 0;
hyperParamsNone.adaptiveNoise = 0;
hyperParamsNone.lambda = 0;
hyperParamsNone.simplePatternMatching = 1;

hyperParamsAll = stdHyperParams;

hyperParamsNoFPFiltering = stdHyperParams;
hyperParamsNoFPFiltering.doFPFiltering = 0;

hyperParamsNoAdaptiveNoise = stdHyperParams;
hyperParamsNoAdaptiveNoise.adaptiveNoise = 0;

hyperParamsNoAngles = stdHyperParams;
hyperParamsNoAngles.lambda = 0;

hyperParamsSimplePatternMatching = stdHyperParams;
hyperParamsSimplePatternMatching.simplePatternMatching = 1;

HYPERS = {hyperParamsNone, hyperParamsAll, hyperParamsNoFPFiltering, hyperParamsNoAdaptiveNoise, hyperParamsNoAngles, hyperParamsSimplePatternMatching};
%%
allPatterns = read_patterns('tracking/datasets/framework');
patterns = zeros(10,4,3);
patternNames = {};
for i=1:length(allPatterns)
    patterns(i,:,:) = allPatterns(i).pattern;
    patternNames{i} = allPatterns(i).name;
end

RESULTS = cell(size(testNames,2), 10);
ERRORS = zeros(size(testNames,2), 10, 4);
NNZ = zeros(size(testNames,2), 10, 4);

dataDirs = cell(10,1);
for i=0:9
   dataDirs{i+1} = strcat('data_difficulty_', string(i));
end
for k=1:size(testNames,2)
    for i=0:9
        curDataName = dataDirs{i+1};
        path = strcat('modernMethods/data/matlab/', curDataName, '.mat');
        load(path);
        clearvars resultStruct;
        resultStruct.name = strcat(testNames{k}, '_', 'Difficulty', string(i));
        for q=0:3
            eval(strcat('D = ', 'D', string(q), ';'));
            eval(strcat('pos = ', 'pos', string(q), ';'));
            eval(strcat('quat = ', 'quat', string(q), ';'));
            
            resultStruct.(strcat('D', string(q))) = D;
            resultStruct.(strcat('pos', string(q))) = pos;
            resultStruct.(strcat('quat', string(q))) = quat;

            N = size(D,1);
            D = permute(D, [2, 1, 3, 4]);
            formattedData = reshape(D, size(D,1), [], 3);
            initialStates = zeros( N, 3+3+4+4);
            for j = 1:size(initialStates,1)
                initialStates(j,1:3) = pos(j, 1, :);
                initialStates(j,4:6) = zeros(1,3);
                initialStates(j,7:10) = quat(j, 1, :);
                initialStates(j, 11:14) = zeros(1,4);
            end

            hyperParams = HYPERS{k};
            quatMotionType = 'brownian';
            [estimatedPositions, estimatedQuats, markerAssignemnts, falsePositives] = ownMOT(formattedData, patterns, patternNames ,initialStates, N, pos, quat, quatMotionType, hyperParams);
            resultStruct.(strcat('estPos', string(q))) = estimatedPositions;
            resultStruct.(strcat('estQuat', string(q))) = estimatedQuats;
            resultStruct.(strcat('markerAssignment', string(q))) = markerAssignemnts;
            resultStruct.(strcat('falsePositives', string(q))) = falsePositives;
            %TODOsave(['resultsKF/', curDataName,'_', testNames{k}, '.mat'], 'estimatedPositions', 'estimatedQuats', 'markerAssignemnts', 'falsePositives')
        
            totalError = performanceVisualization(estimatedPositions, pos, estimatedQuats, quat, patterns, 0);
            numFPs = nnz(isnan(totalError))
            NNZ(k,i+1,q+1) = numFPs;
            avgErr = mean(totalError, 'all', 'omitnan')
            ERRORS(k,i+1,q+1) = avgErr;
            resultStruct.numFPs = numFPs;
            resultStruct.avgError = avgErr;
        end
        %RESULTS(i+1,k) = resultStruct;
    end
end

%save('resultsKF/allResults.mat', 'RESULTS')

%% load data
path = 'modernMethods/data/matlab/generated_data.mat';
load(path);
size(D);
N = size(D,1);
D = permute(D, [2, 1, 3, 4]);
formattedData = reshape(D, size(D,1), [], 3);


%% prepare data and get patterns
initialStates = zeros( N, 3+3+4+4);
for i = 1:size(initialStates,1)
   initialStates(i,1:3) = pos(i, 1, :);
   initialStates(i,4:6) = zeros(1,3);
   initialStates(i,7:10) = quat(i, 1, :);
   initialStates(i, 11:14) = zeros(1,4);
end

allPatterns = read_patterns('tracking/datasets/framework');
patterns = zeros(10,4,3);
patternNames = {};
for i=1:length(allPatterns)
    patterns(i,:,:) = allPatterns(i).pattern;
    patternNames{i} = allPatterns(i).name;
end
    
    
%% test MOT
for i=1:size(testNames, 2)
    hyperParams = HYPERS{i};
    quatMotionType = 'brownian';
    [estimatedPositions, estimatedQuats, markerAssignemnts, falsePositives] = ownMOT(formattedData, patterns, patternNames ,initialStates, N, pos, quat, quatMotionType, hyperParams);
    save(['resultsKF/', curDataName,'_', testNames{i}, '.mat'], 'estimatedPositions', 'estimatedQuats', 'markerAssignemnts', 'falsePositives')
end

%% Evaluate tracking performance 
errs = zeros(size(testNames,2),1);
for i=1:size(testNames,2)
    load(['resultsKF/', curDataName,'_', testNames{i}, '.mat'])
    totalError = performanceVisualization(estimatedPositions, pos, estimatedQuats, quat, patterns, 0);
    fprintf(testNames{i})
    nnz(isnan(totalError))
    avgErr = mean(totalError, 'all', 'omitnan')
    errs(i) = avgErr;
end
%totalError = performanceVisualization(estimatedPositions, pos, estimatedQuats, quat, patterns);
%avgErrorPerBird = mean(totalError, 2);
%avgError = mean(totalError, 'all');


%% Save results
%mkdir('resultsKF')
%save(['resultsKF/', curDataName], 'estimatedPositions', 'estimatedQuats', 'markerAssignemnts', 'falsePositives')

%%
%t0=1;
%vizRes(formattedData(t0:end,:,:), patterns, estimatedPositions(:,t0:end,:), estimatedQuats(:,t0:end,:), 1, pos(:, t0:end, :), quat(:, t0:end, :))
