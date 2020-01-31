function [r] = stateTrans(S)
%STATETRANS Summary of this function goes here
%   Detailed explanation goes here
r = [zeros(3,1); S.v + S.a/2; S.a; zeros(3,1)];
end

