function res = measFunc(S,m)
%MEASFUNC Measurement function of the LG-EKF in homogenous coordinates.
%   Mapping from the state space to the measurement space, which rotates
%   and translates the pattern m.
%   
%   Arguments:
%   @S state of the Kalman filter.
%   @m pattern, [4x3] array
%
%   Returns:
%   @res rotated pattern, [4x3] array

res = S.X*m;
end

