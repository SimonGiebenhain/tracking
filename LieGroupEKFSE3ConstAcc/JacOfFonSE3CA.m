function res = JacOfFonSE3CA(S)
%JACOFFONSE3CA Summary of this function goes here
%   Detailed explanation goes here

%syms e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 e12 real;
%[Xe, ve, ae] = expSE3CAvec([e1;e2;e3;e4;e5;e6], [e7;e8;e9], [e10;e11;e12]);
%f = @(X, v, a) stateTrans(comp(X, a, v, Xe, ve, ae));

%J = jacobian(f, [e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 e12])
if S.motionModel == 0
    j = zeros(6);
    res = Ad(expSE3ACvec(-stateTrans(S))) + j;
elseif S.motionModel == 2
    j = [zeros(3,12);
         zeros(3,6) eye(3) eye(3)/2;
         zeros(3,9) eye(3);
         zeros(3,12)];

    res = Ad(expSE3ACvec(-stateTrans(S))) + j;
else
    'constant Velocity model not yet implemented'
end
 
end

