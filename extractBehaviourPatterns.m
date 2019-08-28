load('trackingResults')
load('rawDetections.mat')
load('D_labeled.mat')
%% prepare data
offset = 130;
formattedData = formattedData(1011+offset-1:end, :,:);

% estimatedPositions, estimatedQuats, VICONPos, VICONquat

%% detect movement, i.e. flying, landing, starting or walking

minLength = 20;
maxGapLength = minLength/2;

isMoving = zeros(10,length(estimatedPositions)-1);
for k=1:10
    movementSpeed = sum(squeeze((estimatedPositions(k, 2:end,:) - estimatedPositions(k, 1:end-1,:)).^2), 2);
    if k == 8
        k
    end
    isMoving(k,:) = movementSpeed > 50;
end

% TODO encapuslate in cleaning method
for k=1:10
    counter = 0;
    for t=1:length(estimatedPositions)-1
        if isMoving(k,t)
            counter = counter + 1;
        elseif counter > 0
            maxt = min(t+maxGapLength, size(isMoving, 2));
            if any(isMoving(k, t:maxt))
                counter = counter + 1;
                isMoving(k,t) = 1;
            else
                if counter >= minLength
                    counter = 0;
                else
                    isMoving(k, t-counter:t) = 0;
                    counter = 0;
                end
            end
        end
    end
    if counter < minLength
        isMoving(k, t-counter:t) = 0;
    end
end

%% identify flying
% TODO movementSpeed higher than 200-300 or smth. and than clean

%% starting and ladnding
% moving with elevation changes

%% identify walking
% moving but not flying landing or starting