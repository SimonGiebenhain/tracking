function S = vecToSE3CAAlgebra(S)
%VECTOSE3CA Mapping from R^12 to Lie algebra of SE(3) x R^3 x R^3
%   Arguments:
%   @S not represented as a regular 12-dim vector. Rather S.X is a 6-dim
%   vector, S.v and S.a are both 3-dim vectors.
%   Returns:
%   S element from the Lie algebra of SE(3) x R^3 x R^3, where S.X is [4x4]
%   matrix and S.v and S.a are both 3-dim vecotrs.
%   Note for R^3 the mapping is the identity, hence only X has to be
%   transformed.
S.X = vecToSE3Algebra(S.X);
end

