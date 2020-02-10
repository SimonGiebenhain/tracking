function res = Ad(S)
%AD Summary of this function goes here
%   Detailed explanation goes here
R = S.X(1:3, 1:3);
t = S.X(1:3, 4);
res = [R zeros(3) zeros(3) zeros(3)
       vecToSO3Algebra(t)*R R zeros(3) zeros(3);
       zeros(3) zeros(3) eye(3) zeros(3);
       zeros(3) zeros(3) zeros(3) eye(3)];
%    res = [R vecToSO3Algebra(t)*R zeros(3) zeros(3);
%        zeros(3) R zeros(3) zeros(3);
%        zeros(3) zeros(3) eye(3) zeros(3);
%        zeros(3) zeros(3) zeros(3) eye(3)];
end

