function [augmentedPositions] = postProcessing(positions, ghostTracks, patterns)
%POSTPROCESSING Summary of this function goes here
%   Detailed explanation goes here
augmentedPositions = positions;
for g=1:length(ghostTracks)
    if ~isempty(ghostTracks{g})
        t0 = ghostTracks{g}.beginningFrame;
        traj = ghostTracks{g}.trajectory(~any(isnan(ghostTracks{g}.trajectory), 2), :);
        T = length(traj);
        trackedBirds = ~isnan(positions(:, t0:t0+T-1, 1));
        %nTrackedBirds = sum(trackedBirds, 1);
        %framesAlmostFull = sum(nTrackedBirds == length(patterns) - 1 - sum(abs(patterns)>=1000, 'all'));
        %if framesAlmostFull > 400
        birdFreq = sum(trackedBirds, 2);
        corruptedPatterns = any(abs(patterns) >= 1000, [2, 3]);
        birdFreq(corruptedPatterns) = Inf;
        
        if sum(birdFreq > 500) == length(patterns) - 1
            if sum(birdFreq < 250) == 1
                patternIdx = find(birdFreq < 250);
                augmentedPositions(patternIdx, t0:t0+T-1, :) = traj; 
            end
        end
        
        %end
    end
end
end

