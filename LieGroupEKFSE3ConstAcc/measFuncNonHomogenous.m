function rot_pat = measFuncNonHomogenous(S, pattern)
%MEASFUNCNONHOMOGENOUS The measurement funciton of the Kalman filter, in
%regular coordinates.
%   The measurement function maps from the state space to the observation
%   space. More concretely it tells where measurements of markers are
%   expected, given the current rotation and position of a bird.
%   Arguments:
%   @S the current state of the LG-EKF object. Where S.X specifies the
%   rotation and translation of the bird.
%   @pattern a [4x3] array specifiying the pattern.
%
%   Returns:
%   @rot_pat a [4x3] array holding the rotated pattern.

%rotation
R = S.X(1:3, 1:3);
%translation
t = S.X(1:3, 4);

rot_pat = (R * pattern')' + t';
end

