function X = vecToSE3Algebra(v)
%VECTOSE3ALGEBRA Summary of this function goes here
%   Detailed explanation goes here
X = [vecToSO3Alegbra(v(1:3)) v(4:6); zeros(1,4)];
end

