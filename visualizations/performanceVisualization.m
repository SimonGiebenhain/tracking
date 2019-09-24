function performanceVisualization(estimatedPositions, positions, estimatedQuats, quats, patterns)
%PERFORMANCEVISUALIZATION Visualize tracking errors of postion and
%orientation for each object to track
%   This function is not finished yet!
%   TODO: Handle partial tracks!
%
%   this function visualizes the euclidean distances of the true position
%   and orientation to the estimated ones for each tracked object.
%   For large number of objects the visualization looks very messy.

    [nObjects, T, ~] = size(estimatedPositions);
    colors = distinguishable_colors(nObjects);
    
    positionDistances = sqrt(sum((estimatedPositions - positions).^2,3));
    
    rotationError = zeros(nObjects, T);
    
    for t = 1:T
        for k = 1:nObjects
            pattern = squeeze( patterns(k,:,:) );
            rotatedPatternEstimation = (Rot(squeeze(estimatedQuats(k,t,:))) * pattern')';
            %rotatedPatternVicon = (  (Rot(squeeze(quats(k,t,:)))) * pattern'  )';
            rotatedPatternVicon = (  (Rot(squeeze(quats(k,t,:)))) * pattern'  )';

            rotationError(k,t) = mean(sqrt(  sum((rotatedPatternEstimation - rotatedPatternVicon).^2,2)));
        end
    end
    
    totalError = zeros(nObjects, T);
    for t = 1:T
        for k = 1:nObjects
            pattern = squeeze( patterns(k,:,:) );
            rotatedPatternEstimation = (Rot(squeeze(estimatedQuats(k,t,:))) * pattern')' + squeeze(estimatedPositions(k,t,:))';
            rotatedPatternVicon = (  (Rot(squeeze(quats(k,t,:)))) * pattern'  )' + squeeze(positions(k,t,:))';

            totalError(k,t) = mean(sqrt(  sum((rotatedPatternEstimation - rotatedPatternVicon).^2,2)));
        end
    end
    
    
%     % Normalize quaternions
%     quatsNormalized = quats./sqrt(sum(quats.^2, 3));
%     estimatedQuatsNormalized = estimatedQuats./sqrt(sum(estimatedQuats.^2, 3));
%     
%     % Since anti-podal unit quaternions represent the same rotation, I take
%     % the samaller distance of the estimated quaternion to the true
%     % quaternions or the anti-podal pair of the true quaternion.
%     quatsDistances = min( sqrt(sum((estimatedQuatsNormalized - quatsNormalized).^2,3)), ...
%                           sqrt(sum((estimatedQuatsNormalized + quatsNormalized).^2,3)));
    
    figure; hold on;
    for i=1:nObjects
        name = sprintf('Object %d', i);
        plot(1:t, totalError(i,:), 'DisplayName', name, 'Color', colors(i,:))
    end
    title('RMSE(Rooted Mean Squared Error)');
    legend;
    xlabel('time')
    ylabel('RMSE between Kalman Filter predictions and corrected VICON predictions')
    hold off;
    
    
    figure;
    subplot(1,2,1)
    hold on;
    for i =1:nObjects
        name = sprintf('Object %d', i);
        plot(1:T, positionDistances(i,:), 'DisplayName', name, 'Color', colors(i,:))
    end
    legend;
    title('Position estimation error');
    hold off;
    
    subplot(1,2,2)
    hold on;
    for i = 1:nObjects
        name = sprintf('Object %d', i);
        %plot(1:T, quatsDistances(i,:), 'DisplayName', name, 'Color', colors(i,:))
        plot(1:T, smoothdata(rotationError(i,:), 'movmedian', 100), 'DisplayName', name, 'Color', colors(i,:))

    end
    legend;
    title('Orientation estimaton error');
    hold off;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure; 
    subplot(1,2,1)
    hold on;
    for k=1:nObjects
        plot(1:T, quats(k,:,1), 'Color', colors(k,:));
    end
    hold off;
    subplot(1,2,2)
    hold on;
    for k=1:nObjects
        plot(1:T, estimatedQuats(k,:,1), 'Color', colors(k,:));
    end
    hold off;
    
    
    figure; 
    subplot(1,2,1)
    hold on;
    for k=1:nObjects
        plot(1:T, quats(k,:,2), 'Color', colors(k,:));
    end
    hold off;
    subplot(1,2,2)
    hold on;
    for k=1:nObjects
        plot(1:T, estimatedQuats(k,:,2), 'Color', colors(k,:));
    end
    hold off;
    
    figure; 
    subplot(1,2,1)
    hold on;
    for k=1:nObjects
        plot(1:T, quats(k,:,3), 'Color', colors(k,:));
    end
    hold off;
    subplot(1,2,2)
    hold on;
    for k=1:nObjects
        plot(1:T, estimatedQuats(k,:,3), 'Color', colors(k,:));
    end
    hold off;
    
    figure; 
    subplot(1,2,1)
    hold on;
    for k=1:nObjects
        plot(1:T, quats(k,:,4), 'Color', colors(k,:));
    end
    hold off;
    subplot(1,2,2)
    hold on;
    for k=1:nObjects
        plot(1:T, estimatedQuats(k,:,4), 'Color', colors(k,:));
    end
    hold off;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    figure; 
    subplot(1,2,1)
    hold on;
    for k=1:nObjects
        plot(1:T, positions(k,:,1), 'Color', colors(k,:));
    end
    hold off;
    subplot(1,2,2)
    hold on;
    for k=1:nObjects
        plot(1:T, estimatedPositions(k,:,1), 'Color', colors(k,:));
    end
    hold off;
    
    figure; 
    subplot(1,2,1)
    hold on;
    for k=1:nObjects
        plot(1:T, positions(k,:,2), 'Color', colors(k,:));
    end
    hold off;
    subplot(1,2,2)
    hold on;
    for k=1:nObjects
        plot(1:T, estimatedPositions(k,:,2), 'Color', colors(k,:));
    end
    hold off;
    
    figure; 
    subplot(1,2,1)
    hold on;
    for k=1:nObjects
        plot(1:T, positions(k,:,3), 'Color', colors(k,:));
    end
    hold off;
    subplot(1,2,2)
    hold on;
    for k=1:nObjects
        plot(1:T, estimatedPositions(k,:,3), 'Color', colors(k,:));
    end
    hold off;
end