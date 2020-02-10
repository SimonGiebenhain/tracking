function rot_pat = measFuncNonHomogenous(S, pattern)
%MEASFUNCNONHOMOGENOUS Summary of this function goes here
%   Detailed explanation goes here

R = S.X(1:3, 1:3);
t = S.X(1:3, 4);

rot_pat = (R * pattern')' + t';


% h1 = measFunc(S,[pattern(1,:)';1]); 
% h2 = measFunc(S,[pattern(2,:)';1]); 
% h3 = measFunc(S,[pattern(3,:)';1]); 
% h4 = measFunc(S,[pattern(4,:)';1]); 
% 
% h = [h1(1:3); h2(1:3); h3(1:3); h4(1:3)];
end

