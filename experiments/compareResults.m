N = size(correctedVICON, 1);
K = 10;
colors = distinguishable_colors(K);

allPatterns = read_patterns('tracking/datasets/framework');
patterns = zeros(10,4,3);
patternNames = {};
for i=1:length(allPatterns)
    patterns(i,:,:) = allPatterns(i).pattern;
    patternNames{i} = allPatterns(i).name;
end

%% load VICON
posVICON = zeros(K,N,3);
quatVICON = zeros(K,N,4);

posVICON(:,:,1) = D_offset(:,5:7:end)';
posVICON(:,:,2) = D_offset(:,6:7:end)';
posVICON(:,:,3) = D_offset(:,7:7:end)';

quatVICON(:,:,1) = D_offset(:,4:7:end)';
quatVICON(:,:,2) = D_offset(:,1:7:end)';
quatVICON(:,:,3) = D_offset(:,2:7:end)';
quatVICON(:,:,4) = D_offset(:,3:7:end)';

quatVICON = quatAbsoluteValue(quatVICON);

%% load correctedVICON
correctedPos = zeros(K,N,3);
correctedQuat = zeros(K,N,4);

correctedPos(:,:,1) = correctedVICON(:,5:7:end)';
correctedPos(:,:,2) = correctedVICON(:,6:7:end)';
correctedPos(:,:,3) = correctedVICON(:,7:7:end)';

correctedQuat(:,:,1) = correctedVICON(:,4:7:end)';
correctedQuat(:,:,2) = correctedVICON(:,1:7:end)';
correctedQuat(:,:,3) = correctedVICON(:,2:7:end)';
correctedQuat(:,:,4) = correctedVICON(:,3:7:end)';

correctedQuat = quatAbsoluteValue(correctedQuat);

%% load kalmanFilter
posKalman = zeros(K,N,3);
quatKalman = zeros(K,N,4);
kalmanFilterPredictions = table2array(kalmanFilterPredictionsNoRMSE);
posKalman(:,:,1) = kalmanFilterPredictions(:,5:7:end)';
posKalman(:,:,2) = kalmanFilterPredictions(:,6:7:end)';
posKalman(:,:,3) = kalmanFilterPredictions(:,7:7:end)';

quatKalman(:,:,1) = kalmanFilterPredictions(:,4:7:end)';
quatKalman(:,:,2) = kalmanFilterPredictions(:,1:7:end)';
quatKalman(:,:,3) = kalmanFilterPredictions(:,2:7:end)';
quatKalman(:,:,4) = kalmanFilterPredictions(:,3:7:end)';

quatKalman = quatAbsoluteValue(quatKalman);

%% plot comparion
%figure;
%plot(1:N, squeeze(correctedPos(1,:,:))); hold on;
%plot(1:N, squeeze(posKalman(1,:,:)));
%hold off;

figure;
plot(1:N, squeeze(correctedQuat(10,:,:))); 
figure;
plot(1:N, squeeze(quatKalman(10,:,:)));


%% plot position difference
figure;
posDiff = sqrt(sum((posKalman - correctedPos).^2,3));
hold on;
for k=1:K
    plot(1:N, posDiff(k,:), 'Color', colors(k,:));
end
hold off;
%% plot quaternion difference
figure;
quatDiff = sqrt(sum((quatVICON - correctedQuat).^2,3));
hold on;
for k=1:K
    plot(1:N,quatDiff(k,:), 'Color', colors(k,:));
end
hold off;

%% plot RMSE
rotationError = zeros(K,N);
for t = 1:N
        for k = 1:K
            pattern = squeeze( patterns(k,:,:) );
            rotatedPatternEstimation = (Rot(squeeze(quatKalman(k,t,:))) * pattern')';
            rotatedPatternVicon = (  (Rot(squeeze(correctedQuat(k,t,:)))) * pattern'  )';

            rotationError(k,t) = mean(sqrt(  sum((rotatedPatternEstimation - rotatedPatternVicon).^2,2)));
        end
end
figure;
hold on;
for k=1:K
    plot(1:N,rotationError(k,:), 'Color', colors(k,:));
end
hold off;



%% functions

function positiveQuats = quatAbsoluteValue(quats)
    positiveQuats = zeros(size(quats));
    for k=1:10
        %quatAbs = sqrt(sum((squeeze(quats(k,:,:)).^2),2));
        quatAbs = quats(k,:,1)';
        positiveQuats(k,quatAbs <= 0,1) = -quats(k,quatAbs < 0,1);
        positiveQuats(k,quatAbs <= 0,2) = -quats(k,quatAbs < 0,2);
        positiveQuats(k,quatAbs <= 0,3) = -quats(k,quatAbs < 0,3);
        positiveQuats(k,quatAbs <= 0,4) = -quats(k,quatAbs < 0,4);

        positiveQuats(k,quatAbs > 0,1) = quats(k,quatAbs > 0,1);
        positiveQuats(k,quatAbs > 0,2) = quats(k,quatAbs > 0,2);
        positiveQuats(k,quatAbs > 0,3) = quats(k,quatAbs > 0,3);
        positiveQuats(k,quatAbs > 0,4) = quats(k,quatAbs > 0,4);
    end
    positiveQuats(isnan(quats)) = NaN;
end
