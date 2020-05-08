function res = Ad(S)
%AD Summary of this function goes here
%   Detailed explanation goes here
R = S.X(1:3, 1:3);
t = S.X(1:3, 4);

if S.motionModel == 0
    res = [R zeros(3);
           vecToSO3Algebra(t)*R R];
elseif S.motionModel == 2
    res = [R zeros(3) zeros(3) zeros(3)
           vecToSO3Algebra(t)*R R zeros(3) zeros(3);
           zeros(3) zeros(3) eye(3) zeros(3);
           zeros(3) zeros(3) zeros(3) eye(3)];
else
    'constant Velocity model not yet implemented'
end
%    res = [R vecToSO3Algebra(t)*R zeros(3) zeros(3);
%        zeros(3) R zeros(3) zeros(3);
%        zeros(3) zeros(3) eye(3) zeros(3);
%        zeros(3) zeros(3) zeros(3) eye(3)];
end

