function kF = switchMotionModel(kF, v, newMotionModel, params)
%SWITCHMOTIONMODEL Summary of this function goes here
%   Detailed explanation goes here

if kF.mu.motionModel == 0 && newMotionModel == 2
    
    %invert prediction step made with the old motion model
    F = JacOfFonSE3CA(kF.mu);
    kF.mu = comp(kF.mu, expSE3ACvec(-stateTrans(kF.mu)));
    kF.P = kF.P - kF.Q;
    %invF = inv(F);
    %s.P = invF * s.P * invF';
    kF.P = F\kF.P/F;
    %s.P = F*s.P*F';
    
    %prepare new motion model, then redo prediction step with new motion
    %model
    kF.Q = diag([repmat(params.quatNoise, 3, 1);
                 repmat(params.posNoise, 3, 1);
                 repmat(params.motNoise, 3, 1);
                 repmat(params.accNoise, 3, 1)]);
    kF.P = [kF.P zeros(6);
            zeros(3,6) 20*eye(3) zeros(3);
            zeros(3,6) zeros(3) 5*eye(3)];
    kF.mu.v = [0; 0; 0]; % TODOTODOTODOTODO: v = 0 oder v = v???
    kF.mu.a = [0; 0; 0];
    kF.mu.motionModel = 2;
    
    % redo prediction step:
    F = JacOfFonSE3CA(kF.mu);
    kF.mu = comp(kF.mu, expSE3ACvec(stateTrans(kF.mu)));
    kF.P = F*kF.P*F' + kF.Q;
    
    kF.framesInNewMotionModel = 0;
    
elseif kF.mu.motionModel == 2 && newMotionModel == 0
    kF.Q = diag([repmat(params.quatNoise, 3, 1);
                 repmat(params.posNoiseBrownian, 3, 1)]);
    kF.P = kF.P(1:6, 1:6);
    kF.mu.v = 0;
    kF.mu.a = 0;
    kF.mu.motionModel = 0;
    
    kF.framesInNewMotionModel = 0;

end
end

