function [H, J] = observation_function(pattern)
%QUAT_TO_MAT Construct rotation matrix from quaternion
%   Using the formula from: https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
%   this function computes a 3x3 rotation matrix from a quaternion with the
%   same rotation properties, i.e. R*x = q*x*q^(-1)
%   The function normalizes the quaternion before applying the formula.

    Rot = quat_to_mat();
   
    syms x y z vx vy vz q1 q2 q3 q4 vq1 vq2 vq3 vq4;
    
    H = reshape( (Rot([q1;q2;q3;q4]) * pattern')' + [x; y; z]', [], 1 );
    J = matlabFunction( jacobian(H, [x; y; z; vx; vy; vz; q1; q2; q3; q4; vq1; vq2; vq3; vq4]) ) ;
    H = @(x) reshape( (Rot(x(2*3+1:2*3+4)) * pattern')' + x(1:3)', [], 1 );
        
end


