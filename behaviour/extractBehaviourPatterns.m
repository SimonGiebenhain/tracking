%% load and prepare
% get estimatedPositions, estimatedQuats, VICONPos, VICONquat
load('trackingResults')
load('rawDetections.mat')
load('D_labeled.mat')

offset = 130;
formattedData = formattedData(1011+offset-1:end, :,:);

nObjects = 10;


%% detect movement, i.e. flying, landing, starting or walking

minLength = 20;
maxGapLength = minLength/2;


movementSpeed = zeros(nObjects,length(estimatedPositions)-1);
isMoving = zeros(nObjects,length(estimatedPositions)-1);
for k=1:nObjects
    movementSpeed(k,:) = sum(squeeze((estimatedPositions(k, 2:end,:) - estimatedPositions(k, 1:end-1,:)).^2), 2);
    isMoving(k,:) = movementSpeed(k,:) > 50;
end

isMoving = cleanData(isMoving, nObjects, minLength, maxGapLength);

%% identify flying
isFlying = movementSpeed > 350;

minLength = 20;
maxGapLength = 5;
isFlying = cleanData(isFlying, nObjects, minLength, maxGapLength);

%% identify starting and ladnding
% rule: moving with elevation changes, but not flying
altitudeChange = zeros(nObjects, length(estimatedPositions) - 1);
for k=1:nObjects
   altitudeChange(k,:) = estimatedPositions(k,2:end,3) - estimatedPositions(k,1:end-1,3); 
end
minLength = 5;
maxGapLength = 3;
isLanding = ~isFlying & isMoving & cleanData(-altitudeChange > 5, nObjects, minLength, maxGapLength);
isLanding = cleanData(isLanding, nObjects, minLength, maxGapLength);
isStarting = ~isFlying & isMoving & cleanData(altitudeChange > 5, nObjects, minLength, maxGapLength);
isStarting = cleanData(isStarting, nObjects, minLength, maxGapLength);
%% identify walking
% moving but not flying landing or starting
isWalking = isMoving & ~isFlying & ~isLanding & ~isStarting;
isWalking = cleanData(isWalking, nObjects, minLength, maxGapLength);

%% identify sitting
% inverse of isMoving
isSitting = ~logical(isMoving);
isSitting = uint8(cleanData(isSitting, nObjects, minLength, maxGapLength));

%% save as numpy
path = 'tracking/behaviour/';
mat2np(double(isFlying), [path 'isFlying.pkl'], 'float64');
mat2np(double(isLanding), [path 'isLanding.pkl'], 'float64');
mat2np(double(isStarting), [path 'isStarting.pkl'], 'float64');
mat2np(double(isWalking), [path 'isWalking.pkl'], 'float64');
mat2np(double(isSitting), [path 'isSitting.pkl'], 'float64');

%% save estimated Position and Orientation as numpy
mat2np(double(estimatedPositions(:,:,1)), [path 'positionsX.pkl'], 'float64');
mat2np(double(estimatedPositions(:,:,2)), [path 'positionsY.pkl'], 'float64');
mat2np(double(estimatedPositions(:,:,3)), [path 'positionsZ.pkl'], 'float64');

mat2np(double(estimatedQuats(:,:,1)), [path, 'quats1.pkl'], 'float64');
mat2np(double(estimatedQuats(:,:,2)), [path, 'quats2.pkl'], 'float64');
mat2np(double(estimatedQuats(:,:,3)), [path, 'quats3.pkl'], 'float64');
mat2np(double(estimatedQuats(:,:,4)), [path, 'quats4.pkl'], 'float64');

%% save VICON positions and orientations as numpy
mat2np(double(D_offset), [path, 'vicon.pkl'], 'float64');

%% save detections as numpy
mat2np(double(formattedData(1:size(D_offset,1),:,1)), [path, 'detectionsX.pkl'], 'float64');
mat2np(double(formattedData(1:size(D_offset,1),:,2)), [path, 'detectionsY.pkl'], 'float64');
mat2np(double(formattedData(1:size(D_offset,1),:,3)), [path, 'detectionsZ.pkl'], 'float64');





%% utility functions

function data = cleanData(data, nObjects, minLength, maxGapLength)

assert(size(data,1) == nObjects)
assert(length(size(data)) == 2)

for k=1:nObjects
    counter = 0;
    for t=1:size(data,2)
        if data(k,t)
            counter = counter + 1;
        elseif counter > 0
            maxt = min(t+maxGapLength, size(data, 2));
            if any(data(k, t:maxt))
                counter = counter + 1;
                data(k,t) = 1;
            else
                if counter >= minLength
                    counter = 0;
                else
                    data(k, t-counter:t) = 0;
                    counter = 0;
                end
            end
        end
    end
    if counter < minLength
        data(k, t-counter:t) = 0;
    end
end
end