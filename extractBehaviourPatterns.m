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
% TODO movementSpeed higher than 200-300 or smth. and than clean
isFlying = movementSpeed > 250;

minLength = 20;
maxGapLength = 5;
isFlying = cleanData(isFlying, nObjects, minLength, maxGapLength);

%% starting and ladnding
% moving with elevation changes

%% identify walking
% moving but not flying landing or starting

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