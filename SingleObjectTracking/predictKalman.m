function s = predictKalman(s)
%PREDICTKALMAN Summary of this function goes here
%   Detailed explanation goes here

% Prediction for state vector and covariance:
if strcmp(s.type, 'LG-EKF')
    F = JacOfFonSE3CA(s.mu);
    s.mu = comp(s.mu, expSE3ACvec(stateTrans(s.mu)));
    s.P = F*s.P*F' + s.Q;
else
    s.x = s.A*s.x;
    s.P = s.A * s.P * s.A' + s.Q;
end



