function X = expSE3vec(v)
%EXPSE3VEC Exponential map, linking Lie Group SE(3) and its Lie algebra
%   Normally maps form the Lie algebra to the Lie group. This
%   implementation however directly maps from the R^3 to the Lie group, as
%   R^6 and the Lie algebra of SE(3) are isomorphic. I.e. this
%   implementation works with a vector representation of the elements from
%   the Lie algebra, instead of the matrix representation.
%   Formula taken from: http://www.ethaneade.org/lie.pdf
phi = v(1:3);
t = v(4:6);
n = norm(phi);
if n == 0
    R = eye(3);
    V = eye(3);
else
    a = sin(n)/n;
    b = (1-cos(n))/n^2;
    c = (1-a)/n^2;
    w = vecToSO3Algebra(phi);
    R = eye(3) + a*w + b*w^2;
    V = eye(3) + b*w + c*w^2;
end

X = [R V*t; zeros(1,3) 1];
end

