function [S] = expSE3ACvec(s)
%EXPSE3CAVEC Summary of this function goes here
%   Detailed explanation goes here
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

