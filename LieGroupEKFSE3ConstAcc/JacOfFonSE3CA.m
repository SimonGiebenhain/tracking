function res = JacOfFonSE3CA(S)
%JACOFFONSE3CA Calculates the Jacobian of the state transition function for
% state S.
%   The Extended Kalman FIlter linearizes the state transition function
%   around its current estimate for some computations. This function
%   implements the formula form the LG-EKF paper https://hal.archives-ouvertes.fr/hal-00903252/document
%   Arguemnts:
%   @S state of the LG-EKF
%   
%   Returns:
%   @res Jacobian of the state transition function.
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

