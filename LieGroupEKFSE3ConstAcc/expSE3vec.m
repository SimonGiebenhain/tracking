function X = expSE3vec(v)
%EXPSE3VEC Summary of this function goes here
%   Detailed explanation goes here
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
%L = eye(3) + ((n-sin(n))/n^3)*(phi*phi') + (1-cos(n))/n^2*vecToSO3Algebra(phi);
%X = [expSO3vec(phi) L*t; zeros(1,3) 1];
X = [R V*t; zeros(1,3) 1];
end

