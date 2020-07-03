function X = vecToSO3Algebra(v)
%VEC2LIEALGEBRA Mapping from R^3 to the Lie Algebra of SO(3)
X = [0 -v(3) v(2); 
     v(3) 0 -v(1); 
     -v(2) v(1) 0];
end

