%% load
load('rawDetections.mat')
load('trackingResults')
load('D_labeled.mat')

offset = 130;

formattedData = formattedData(1011+offset-1:end, :,:);

D_offset = D_labeled(offset-1:end,:);
VICONX = D_offset(:,5:7:end);
VICONY = D_offset(:,6:7:end);
VICONZ = D_offset(:,7:7:end);
VICONq2 = D_offset(:,1:7:end);
VICONq3 = D_offset(:,2:7:end);
VICONq4 = D_offset(:,3:7:end);
VICONq1 = D_offset(:,4:7:end);

VICONpos = cat(3, VICONX, VICONY, VICONZ);
VICONpos = permute(VICONpos, [2 1 3]);

VICONquat = cat(3, VICONq1, VICONq2, VICONq3, VICONq4);
VICONquat = permute(VICONquat, [2 1 3]);

if size(VICONpos,2) < size(estimatedPositions,2)
    estimatedPositions = estimatedPositions(:,1:size(VICONpos,2),:);
    estimatedQuats = estimatedQuats(:, 1:size(VICONquat,2), :);
else
    VICONpos = VICONpos(:,1:size(estimatedPositions,2),:);
    VICONquat = VICONquat(:,1:size(estimatedPositions,2),:);
end
%%

T = size(VICONpos,2);

correctedVICONpos = NaN * zeros(size(VICONpos));
correctedVICONquat = NaN * zeros(size(VICONquat));

for k=1:10
    for t=1:T
        pos = squeeze(VICONpos(k,t,:));
        if ~any(isnan(pos))
            birds = squeeze(estimatedPositions(:,t,:));
            dist = pdist2(reshape(pos, 1, 3), birds);
            [minDist, bird] = min(dist);
            if sum(correctedVICONpos(bird,t,:)) > 0
                fprintf('hi')
            end
            correctedVICONpos(bird,t,:) = pos;
            correctedVICONquat(bird,t,:) = VICONquat(k,t,:);  
        end
    end
end

%%
performanceVisualization(estimatedPositions, correctedVICONpos, estimatedQuats, correctedVICONquat, patterns);
%%
t0=1;
vizRes(formattedData(t0:3000,:,:), patterns, estimatedPositions(:,t0:end,:), estimatedQuats(:,t0:end,:), 1, correctedVICONpos(:,t0:end,:), correctedVICONquat(:,t0:end,:))
