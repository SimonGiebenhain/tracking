%% Parameter settings to test

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
stdHyperParams.quatNoise = 0.5;
stdHyperParams.quatMotionNoise = 0.05;
stdHyperParams.measurementNoise = 100;
stdHyperParams.certaintyScale = 3;

testNames = {'none', 'onlyFPFiltering', 'simplePatternMatching', 'all', 'noAdaptiveNoise', };
hyperParamsNone = stdHyperParams;
hyperParamsNone.doFPFiltering = 0;
hyperParamsNone.adaptiveNoise = 0;
hyperParamsNone.lambda = 0;
hyperParamsNone.simplePatternMatching = 1;

hyperParamsOnlyFP = hyperParamsNone;
hyperParamsOnlyFP.doFPFiltering = 1;

hyperParamsAll = stdHyperParams;

hyperParamsNoAdaptiveNoise = stdHyperParams;
hyperParamsNoAdaptiveNoise.adaptiveNoise = 0;

hyperParamsSimplePatternMatching = stdHyperParams;
hyperParamsSimplePatternMatching.simplePatternMatching = 1;

HYPERS = {hyperParamsNone, hyperParamsOnlyFP, hyperParamsSimplePatternMatching, hyperParamsAll, hyperParamsNoAdaptiveNoise};
%%
allPatterns = read_patterns('tracking/datasets/framework');
patterns = zeros(10,4,3);
patternNames = {};
for i=1:length(allPatterns)
    patterns(i,:,:) = allPatterns(i).pattern;
    patternNames{i} = allPatterns(i).name;
end

RESULTS = cell(size(testNames,2), 13);
ERRORS = zeros(size(testNames,2), 13);
NUMLOSTTRACKS= zeros(size(testNames,2), 13);

dataDirs = cell(13,1);
for i=0:12
   dataDirs{i+1} = strcat('data_difficultyFINAL_', string(i));
end
for k=4:size(testNames,2)
    for i=10:12
        [k,i]
        curDataName = dataDirs{i+1};
        path = strcat('modernMethods/data/matlab/', curDataName, '.mat');
        load(path);
        clearvars resultStruct;
        resultStruct.name = strcat(testNames{k}, '_', 'Difficulty', string(i));
        avgErrs = 0;
        lostTracks = zeros(4,1);
        totalNumFrames = 0;
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
            [estimatedPositions, estimatedQuats, markerAssignemnts, falsePositives] = ownMOT(formattedData, patterns, patternNames , 1, initialStates, N, 0, pos, quat, quatMotionType, hyperParams);
            resultStruct.(strcat('estPos', string(q))) = estimatedPositions;
            resultStruct.(strcat('estQuat', string(q))) = estimatedQuats;
            resultStruct.(strcat('markerAssignment', string(q))) = markerAssignemnts;
            resultStruct.(strcat('falsePositives', string(q))) = falsePositives;
        
            totalError = performanceVisualization(estimatedPositions, pos, estimatedQuats, quat, patterns, 0);
            numLostTracks = nnz(isnan(totalError));
            lostTracks(q+1) = numLostTracks;
            err = mean(totalError, 'all', 'omitnan')
            avgErrs = avgErrs + err*size(totalError,2);
            totalNumFrames = totalNumFrames + size(totalError,2);
        end
        avgErrs = avgErrs / totalNumFrames; 
        avgErrs
        totalLostTracks = sum(lostTracks);
        resultStruct.avgErr = avgErrs;
        resultStruct.numLostTracks = totalLostTracks;
        RESULTS{i+1,k} = resultStruct;
    end
end

save('resultsKF/allResultsFINALFINALGOOD.mat', 'RESULTS')

%%
evalKFasSOT('artificial_higher.mat', 6);

%%evalKFasSOT('artificial_medium.mat', 6);
evalKFasSOT('artificial_none.mat', 4);
evalKFasSOT('birdlike_higher.mat', 6);
evalKFasSOT('birdlike_medium.mat', 6);
evalKFasSOT('birdlike_none.mat', 4);
%%

load('resultsKF/allResultsFINALFINALOLD.mat');
%%
errs = zeros(13, 5);
lostTracks = zeros(13, 5);
for k=1:13
    for m=1:5
        errs(k, m) = RESULTS{k, m}.avgErr;
        lostTracks(k, m) = RESULTS{k, m}.numLostTracks;
    end
end
%figure; hold on;
%for i=1:6
%    plot(errs, 'DisplayName', testNames{i})
%end
%legend; hold off;
logErrs = log(1+errs);
symbols = {'o', '*', 's', 'd', '^'};
figure; hold on;
for i=1:5
    plot(logErrs(:, i), ['--', symbols{i}])
end
legend(testNames{1,1}, testNames{1,2}, testNames{1,3}, testNames{1,4}, testNames{1,5})
xlabel('noise level')
ylabel('log(average pose error + 1)')
hold off;

logLostTracks = log(lostTracks+1);
figure; hold on;
for i=1:5
    plot(logLostTracks(:, i), ['--', symbols{i}])
end
legend(testNames{1}, testNames{2}, testNames{3}, testNames{4}, testNames{5})
xlabel('noise level')
ylabel('log(number of lost tracks + 1)')
hold off;

%% Calc ID-switches
idsw = zeros(13, 5);
for k=1:13
    for m=1:5
        res = RESULTS{k,m};
        idsw(k, m) = calcIDsw(res.pos0, res.estPos0);
        idsw(k, m) = idsw(k, m) + calcIDsw(res.pos1, res.estPos1);
        idsw(k, m) = idsw(k, m) + calcIDsw(res.pos2, res.estPos2);
        idsw(k, m) = idsw(k, m) + calcIDsw(res.pos3, res.estPos3);
    end
end
%%
idsw = floor(idsw);
logIDsw = log(idsw+1);
figure; hold on;
for i=1:5
    plot(logIDsw(:, i), ['--', symbols{i}])
end
legend(testNames{1}, testNames{2}, testNames{3}, testNames{4}, testNames{5})
xlabel('noise level')
ylabel('log(number of ID-switches + 1)')
hold off;

%% Calc MOTP
motp = zeros(13, 5);
for k=1:13
    for m=1:5
        res = RESULTS{k,m};
        [eP, eQ] = removeIDswitches(res.pos0, res.estPos0, res.estQuat0);
        poseError =  performanceVisualization(eP, res.pos0, eQ, res.quat0, patterns, 0);
        motp(k,m) = mean(poseError, 'all', 'omitnan')*size(res.pos0,2);
        totalFrames = size(res.pos0,2);
        
        [eP, eQ] = removeIDswitches(res.pos1, res.estPos1, res.estQuat1);
        poseError =  performanceVisualization(eP, res.pos1, eQ, res.quat1, patterns, 0);
        motp(k,m) = motp(k,m) +  mean(poseError, 'all', 'omitnan')*size(res.pos1,2);
        totalFrames = totalFrames + size(res.pos1,2);
        
        [eP, eQ] = removeIDswitches(res.pos2, res.estPos2, res.estQuat2);
        poseError =  performanceVisualization(eP, res.pos2, eQ, res.quat2, patterns, 0);
        motp(k,m) = motp(k,m) +  mean(poseError, 'all', 'omitnan')*size(res.pos2,2);
        totalFrames = totalFrames + size(res.pos2,2);
        
        [eP, eQ] = removeIDswitches(res.pos3, res.estPos3, res.estQuat3);
        poseError =  performanceVisualization(eP, res.pos3, eQ, res.quat3, patterns, 0);
        motp(k,m) = motp(k,m) +  mean(poseError, 'all', 'omitnan')*size(res.pos3,2);
        totalFrames = totalFrames + size(res.pos3,2);
        motp(k,m) = motp(k,m) / totalFrames;
    end
end

%% Calc MOTA
T = size(pos0,2) + size(pos1,2) + size(pos2,2) + size(pos3,2);
mota = 1 - (lostTracks + idsw)/(10*T)

%% plot MOTA and MOTP
figure; hold on;
for i=1:5
    plot(mota(:, i), ['--', symbols{i}])
end
legend(testNames{1}, testNames{2}, testNames{3}, testNames{4}, testNames{5})
xlabel('noise level')
ylabel('MOTA')
hold off;

figure; hold on;
for i=1:5
    plot(motp(:, i), ['--', symbols{i}])
end
legend(testNames{1}, testNames{2}, testNames{3}, testNames{4}, testNames{5})
xlabel('noise level')
ylabel('Pose-MOTP')
hold off;
%%
function nIDsw = calcIDsw(pos, estPos)
    nIDsw = 0;
    T = size(pos, 2);
    for t=1:T
        p = squeeze(pos(:, t, :));
        eP = squeeze(estPos(:, t, :));
        nans = isnan(eP(:, 1));
        eP(nans, :) = [];
        p(nans, :) = [];
        dist = pdist2(p, eP);
        %dist = dist + eye(size(dist))*1000000;
        [~, minDist] = min(dist, [], 2);
        %minDist should be 1, ..., 10
        nIDsw = nIDsw + nnz(minDist - linspace(1, length(p), length(p))')/2;
    end
end

function [estPos, estQ] = removeIDswitches(pos, estPos, estQ)
    T = size(pos, 2);
    for t=1:T
       dist = pdist2(squeeze(pos(:, t,:)), squeeze(estPos(:,t,:)));
       for i=1:10
          [~, mini] = min(dist(i, :));
          if mini ~= i
             estPos(i, t, :) = NaN; 
             estQ(i, t, :) = NaN;
          end
       end
    end
end


function evalKFasSOT(data_name, numDets)
    data_name
    if strcmp(data_name, 'birdlike_none.mat')
        load('tracking/modernMethods/data/birdlike_higher.mat')
    end
    load(['tracking/modernMethods/data/', data_name])

    [T, N, ~] = size(pos);
    dets = zeros(T, N, numDets*3) * NaN;

    detections = permute(reshape(detections, [T, N, 4, numDets]), [1, 2, 4, 3]);
    lostIdx = detections(:, :, :, 4) == 1;
    lostIdx = repmat(lostIdx, [1, 1, 1, 3]);
    D = detections(:, :, :, 1:3);
    D(lostIdx) = NaN;
    
    stdHyperParams.doFPFiltering = 0;
    stdHyperParams.adaptiveNoise = 1;
    stdHyperParams.lambda = 0;
    stdHyperParams.simplePatternMatching = 0;

    stdHyperParams.costOfNonAsDtTA = 1;
    stdHyperParams.certaintyFactor = 50;
    stdHyperParams.useAssignmentLength = 1;
    stdHyperParams.whenFPFilter = 0.8;
    stdHyperParams.thresholdFPFilter = 0.7;
    stdHyperParams.costOfNonAsDtMA = 0.095;
    stdHyperParams.eucDistWeight = 1/8;
    stdHyperParams.posNoise = 0.25;
    stdHyperParams.motNoise = 0.25;
    stdHyperParams.quatNoise = 0.075;
    stdHyperParams.quatMotionNoise = 0.05;
    stdHyperParams.measurementNoise = 2;
    stdHyperParams.certaintyScale = 0.3;
    quatMotionType = 'brownian';
    
    predPos = zeros(250, T, 3);
    predQuats = zeros(250, T, 4);
    for n=1:250
        if length(size(patterns)) == 4
            pats = reshape(patterns(1, n, :, :), 1, 4, 3);
        else
            pats = reshape(patterns(n, :, :), 1, 4, 3);
        end
        p = squeeze(pos(:, n, :));
        q = squeeze(quats(:, n, :));
        initialStates = [p(1, :) 0 0 0 rotm2quat(reshape(q(1, :), 3, 3)') 0 0 0 0];
        [estimatedPositions, estimatedQuats] = ownMOT(squeeze(D(:, n, :, :)), pats, {'pat'} ,1 , initialStates, 1, 0, -1, -1, quatMotionType, stdHyperParams);
        predPos(n, :, :) = squeeze(estimatedPositions);
        predQuats(n, :, :) = squeeze(estimatedQuats);
    end
    
    save(['test_KFresult_', data_name], 'predPos', 'predQuats')

end

