function kF = switchMotionModel(kF, v, newMotionModel, params)
%SWITCHMOTIONMODEL Switches the motion model of a Kalman filter
%   Arguemnts:
%   @kF struct holding all information about the Kalman filter
%   @v TODO
%   @newMotionModel integer representing the type of motion model. Here 0
%   stands for a brwonian motion model, 1 for a constant velocity model and
%   2 for a constant acceleration model. Note that a constant velocity
%   model is not implemented yet. I'm only switching back and forth between
%   0 and 2.
%   @params struct containing noise assumptions of the system, which are 
%   necessary to specify for a Kalman filter.
%
%   Returns:
%   @kF updated Kalman filter struct

if kF.mu.motionModel == 0 && newMotionModel == 2
    
    %invert prediction step made with the old motion model
    F = JacOfFonSE3CA(kF.mu);
    kF.mu = comp(kF.mu, expSE3ACvec(-stateTrans(kF.mu)));
    kF.P = kF.P - kF.Q;
    kF.P = F\kF.P/F;
    
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
    %TODO: should I undo prediction step of constant acceleration motion
    %model?
    kF.Q = diag([repmat(params.quatNoise, 3, 1);
                 repmat(params.posNoiseBrownian, 3, 1)]);
    kF.P = kF.P(1:6, 1:6);
    kF.mu.v = 0;
    kF.mu.a = 0;
    kF.mu.motionModel = 0;
    
    kF.framesInNewMotionModel = 0;

% All other transitions not implemented
else
    err_msg = ['Switching from motion model ' , num2str(kf.mu.motionModel), ...
               ' to motion model ', num2str(newMotionModel), ...
               ' is not implemented!\n'];
    help_msg = 'Note that in only motion model 0 and 2 are implemented!';
    error([err_msg, help_msg])
end
end

