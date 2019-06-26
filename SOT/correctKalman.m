function s = correctKalman(s, detectionsIdx)
%CORRECTKALMAN Summary of this function goes here
%   Detailed explanation goes here

% Correction based on observation
J = s.J;
subindex = @(A, rows) A(rows, :);     % for row sub indexing
J = @(x) subindex(J(x(7), x(8), x(9), x(10)), detectionsIdx);

%Evaluate Jacobian at current position
J = J(s.x);
% Compute Kalman gain factor:
K = s.P*J'/(J*s.P*J'+s.R);

z = s.H(s.x);
s.x = s.x + K*(s.z-z(detectionsIdx));

% TODO immer machen? oder nur wenn detection?
% Correct covariance matrix estimate
s.P = s.P - K*J*s.P;

end

