function [S] = expSE3ACvec(s)
%EXPSE3CAVEC Summary of this function goes here
%   Detailed explanation goes here
S.X = expSE3vec(s(1:6));
S.v = s(7:9);
S.a = s(10:12);
end

