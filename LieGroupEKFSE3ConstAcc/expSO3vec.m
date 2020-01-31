function res = expSO3vec(e)
    n = norm(e);
    res = eye(3) + (1-cos(n))/n^2*vecToSO3Algebra(e)^2 + sin(n)/n * vecToSO3Algebra(e);
end

