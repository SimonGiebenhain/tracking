function kF = switchGhostMM(kF, newMotionModel, params)
%SWITCHMOTIONMODEL Summary of this function goes here
%   Detailed explanation goes here

if kF.motionModel == 0 && newMotionModel == 2
    
    %invert prediction step made with the old motion model
    kF.x = kF.F\kF.x;
    kF.P = kF.P - kF.Q;
    kF.P = kF.F\kF.P/kF.F;
    
    %prepare new motion model, then redo prediction step with new motion
    %model
    kF.Q = eye(9) .* repelem([50; 5; 1], 3);

    kF.P = [kF.P zeros(3,6);
            zeros(3) 5*eye(3) zeros(3);
            zeros(3) zeros(3) 2*eye(3)];
    kF.x = [kF.x; zeros(6,1)];
    kF.H = [eye(3) zeros(3,6)];
    kF.F = [eye(3) eye(3) 1/2*eye(3);
            zeros(3) eye(3) eye(3);
            zeros(3) zeros(3) eye(3)];
    kF.motionModel = 2;
    
    % redo prediction step:
    kF.x = kF.F*kF.x;
    kF.P = kF.F*kF.P*kF.F' + kF.Q;
    
    kF.framesInMotionModel = 0;
    
elseif kF.motionModel == 2 && newMotionModel == 0
    kF.Q = eye(3) .* repelem(params.processNoise.position, 3);
    kF.P = kF.P(1:3, 1:3);
    kF.x = kF.x(1:3);
    kF.F = eye(3);
    kF.H = eye(3);
    kF.motionModel = 0;
    kF.framesInMotionModel = 0;
end
end



