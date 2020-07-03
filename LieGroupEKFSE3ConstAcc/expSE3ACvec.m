function [S] = expSE3ACvec(s)
%EXPSE3CAVEC Exponential map of SE(3) (x R^3 x R^3), mapping from Euclidean
%space to a Lie group. (normally maps from Lie algebra but this
%implementation prepends the isomorphism between Lie algebra and Euclidean
%space)
%   Arguemnts:
%   @s 6, 9 or 12-dim vector, which is then mapped an element of the Lie
%   Group. When dim==6 then a brownian motion model is currently used, when
%   dim==9 a constant velocity model is used, when dim==12 a constant
%   acceleration model is used.
%
%   Returns:
%   @S element of the (matrix) Lie Group
d = length(s);
if d == 6
    S.X = expSE3vec(s(1:6));
    S.motionModel = 0;
elseif d == 9
    S.X = expSE3vec(s(1:6));
    S.v = s(7:9);
    S.motionModel = 1;
elseif d == 12
    S.X = expSE3vec(s(1:6));
    S.v = s(7:9);
    S.a = s(10:12);
    S.motionModel = 2;
else
   'unexpected motion model' 
end
end

