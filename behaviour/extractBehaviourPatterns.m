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
isSitting = ~isMoving;
isSitting = cleanData(isSitting, nObjects, minLength, maxGapLength);

%% save as numpy
path = 'tracking/behaviour/';
mat2np(uint8(isFlying), [path 'isFlying.pkl'], 'uint8');
mat2np(uint8(isLanding), [path 'isLanding.pkl'], 'uint8');
mat2np(uint8(isStarting), [path 'isStarting.pkl'], 'uint8');
mat2np(uint8(isWalking), [path 'isWalking.pkl'], 'uint8');
mat2np(uint8(isSitting), [path 'isSitting.pkl'], 'uint8');





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