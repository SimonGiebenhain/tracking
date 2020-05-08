function [r] = stateTrans(S)
%STATETRANS Summary of this function goes here
%   Detailed explanation goes here
if S.motionModel == 0
    r = [zeros(3,1); 
         zeros(3,1)];
elseif S.motionModel == 1
    r = [zeros(3,1); 
         S.v; 
         zeros(3,1)];
elseif S.motionModel == 2
    r = [zeros(3,1); 
         S.v + S.a/2; 
         S.a; 
         zeros(3,1)];
else
    'somthing wrong with motionModel'
end

