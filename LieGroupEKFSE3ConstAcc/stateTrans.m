function [r] = stateTrans(S)
%STATETRANS State transion function fot LG-EKF
%   Arguemnts:
%   @S the state of the LG-EKF. S.motionModel indicates which motion model
%   is currently used. S.v and S.a are the current velocity and
%   acceleration accordingly.
%   Note that for the rotation a brownian motion model is assumed, which
%   implies that the rotation does not change during the state transition.
if S.motionModel == 0
    r = [zeros(3,1); 
         zeros(3,1)];
elseif S.motionModel == 1
    r = [zeros(3,1); 
         S.v; 
         zeros(3,1)];
elseif S.motionModel == 2
    r = [zeros(3,1); 
         S.v + S.a/2; 
         S.a; 
         zeros(3,1)];
else
    error(['Somthing wrong with motionModel. MotionModel was set to: ', num2str(S.motionModel)])
end

