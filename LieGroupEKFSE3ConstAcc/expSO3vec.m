function res = expSO3vec(e)
% EXPSO3VEC Exponential map of Lie Group SO(3)
%   The exponential map, maps from the Lie algebra to its Lie group.
%   This implementation works with the vector representation of elements
%   from the Lie algrbea.
    n = norm(e);
    res = eye(3) + (1-cos(n))/n^2*vecToSO3Algebra(e)^2 + sin(n)/n * vecToSO3Algebra(e);
end

