function S = comp(S1, S2)
%COMP Summary of this function goes here
%   Detailed explanation goes here
%X = X1*X2;
%v = v1+v2;
%a = a1+a2;
S.X = S1.X*S2.X;
S.motionModel = S1.motionModel;
if S.motionModel > 0
    S.v=S1.v+S2.v;
end
if S.motionModel == 2
    S.a=S1.a+S2.a;
end
end

