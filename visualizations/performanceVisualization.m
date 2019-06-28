function performanceVisualization(estimatedPositions, positions, estimatedQuats, quats)
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
    
    % Normalize quaternions
    quatsNormalized = quats./sqrt(sum(quats.^2, 3));
    estimatedQuatsNormalized = estimatedQuats./sqrt(sum(estimatedQuats.^2, 3));
    
    % Since anti-podal unit quaternions represent the same rotation, I take
    % the samaller distance of the estimated quaternion to the true
    % quaternions or the anti-podal pair of the true quaternion.
    quatsDistances = min( sqrt(sum((estimatedQuatsNormalized - quatsNormalized).^2,3)), ...
                          sqrt(sum((estimatedQuatsNormalized + quatsNormalized).^2,3)));
    
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
        plot(1:T, quatsDistances(i,:), 'DisplayName', name, 'Color', colors(i,:))
    end
    legend;
    title('Orientation estimaton error');
    hold off;
end