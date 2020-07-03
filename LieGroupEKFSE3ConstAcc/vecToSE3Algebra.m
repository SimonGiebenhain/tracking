function X = vecToSE3Algebra(v)
%VECTOSE3ALGEBRA Mapping from R^6 to the Lie algebra of SE(3)
X = [vecToSO3Alegbra(v(1:3)) v(4:6); zeros(1,4)];
end

