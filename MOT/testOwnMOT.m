%% Generate Data
T = 1000;
nMarkers = 4;
nObjects = 2;
timeDomain = (1:T)';

positions = zeros(nObjects,T,3);
quats = zeros(nObjects,T,4);

positions(1,:,:) = [timeDomain/T*5.*sin(timeDomain/32) cos(timeDomain/68).^2*5 sin(timeDomain/20).^2.*timeDomain/T*3];
quats(1,:,:) = [sin(timeDomain/100) -cos(timeDomain/50) sin(timeDomain/60) cos(timeDomain/30).^2];

positions(2,:,:) = squeeze(positions(1,:,:)) + [sin(timeDomain/60)+0.5,cos(timeDomain/100).^2,zeros(T,1)];
quats(2,:,:) = [cos(timeDomain/60).^2 -cos(timeDomain/50) sin(timeDomain/40).^2 cos(timeDomain/30).^2];

patterns = zeros(nObjects,nMarkers,3);
patterns(1,:,:) = 0.5 * [1      1     1; 
                   0      0     0; 
                   1      0    -2; 
                   -0.5  -2     0.5];

patterns(2,:,:) = 0.5 * [-1     1     0; 
                   -0.4   0.2   1; 
                   1.2    0.5  -1.5; 
                   0      2     0.5];
               
D = zeros(T,nObjects*nMarkers,3);

frameDropRate = 0.2;
markerDropRate = 0.3;
markerNoise = 0.005;

for t=1:T
    for i = 1:nObjects
        % In some frames drop all detections.
        if rand > frameDropRate
            % In some frames drop some individual markers
            missedDetectionsSimple = rand(nMarkers,1) < markerDropRate;
            missedDetections = repmat(missedDetectionsSimple, 3,1);

            % TODO handle cases when only 1 marker was detected per in a frame
            % For now: Make sure there are at least two markers detected
            if sum(~missedDetections) < 6 && nMarkers > 1
                in = randi(4);
                missedDetectionsSimple(in) = 0;
                missedDetections(in) = 0;
                missedDetections(in+4) = 0;
                missedDetections(in+8) = 0;
                
                missedDetectionsSimple( mod(in,4)+1 ) = 0;
                missedDetections( mod(in,4)+1 ) = 0;
                missedDetections( mod(in,4)+1+4 ) = 0;
                missedDetections( mod(in,4)+1+8 ) = 0;
            end

            % Delete some rows in H and R to accomodate for the missed
            % measurements.
            Rot = quatToMat();
            Rot = Rot(quats(i,t,:));
            
            z = (Rot*squeeze(patterns(i,:,:))')' + squeeze(positions(i,t,:))';
            z = z(~missedDetectionsSimple, :);
            noise = reshape( ... 
                mvnrnd(zeros(sum(~missedDetections),1), markerNoise*eye(sum(~missedDetections)),1)',...
                sum(~missedDetectionsSimple),3); % add noise
            z = z + noise;
            detections = NaN*ones(nMarkers, 3);
            detections(~missedDetectionsSimple,:) = z;
        else
            % dropped frames don't have any detections
            detections = NaN * ones(nMarkers,3);
        end
        D(t, (i-1)*nMarkers+1:i*nMarkers, :) = detections;
    end
end

initialStates = zeros(nObjects, 2*3+2*4);
for i = 1:nObjects
   initialStates(i,:) = [reshape(positions(i,1,:), 3,1); zeros(3,1); reshape(quats(i,1,:), 4,1); zeros(4,1)];
end
%% test MOT
[estimatedPositions, estimatedQuats] = ownMOT(D, patterns, initialStates, positions);

%%
performanceVisualization(estimatedPositions, positions, estimatedQuats, quats);
