function [Rot, JRow1, JRow2, JRow3] = quat_to_mat(q, normalize)
%QUAT_TO_MAT Construct rotation matrix from quaternion
%   Using the formula from: https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
%   this function computes a 3x3 rotation matrix from a quaternion with the
%   same rotation properties, i.e. R*x = q*x*q^(-1)
%   The function normalizes the quaternion before applying the formula.

if ~exist('normalize','var') && ~exist('q', 'var')
    normalize = 1;
end
    syms q1 q2 q3 q4;
    
    norm_const = q1^2+q2^2+q3^3+q4^2;
    
    R = eye(3) + 1/norm_const * ...
        [-q3^2-q4^2         q2*q3-q4*q1   q2*q4+q3*q1;
        q2*q3+ q4*q1   -q2^2- q4^2       q3*q4-q2*q1;
        q2*q4-q3*q1    q3*q4+q2*q1   -q2^2-q3^2      ];
    
if normalize
    Rot = @(q) eye(3) + 1/sum(q.^2) * ...
        [-q(3)^2-q(4)^2         q(2)*q(3)-q(4)*q(1)   q(2)*q(4)+q(3)*q(1);
        q(2)*q(3)+ q(4)*q(1)   -q(2)^2- q(4)^2       q(3)*q(4)-q(2)*q(1);
        q(2)*q(4)-q(3)*q(1)    q(3)*q(4)+q(2)*q(1)   -q(2)^2-q(3)^2      ];
    
    syms q1 q2 q3 q4;
        
    RRow1 = [1 0 0] + 1/(q1^2+q2^2+q3^3+q4^2) * ...
                        [-q3^2-q4^2     q2*q3-q4*q1   q2*q4+q3*q1];
                    
    RRow2 = [0 1 0] + 1/(q1^2+q2^2+q3^3+q4^2) * ...
                        [q2*q3+ q4*q1   -q2^2- q4^2   q3*q4-q2*q1];
                        
    RRow3 = [0 0 1] + 1/(q1^2+q2^2+q3^3+q4^2) * ...
                        [q2*q4-q3*q1    q3*q4+q2*q1   -q2^2-q3^2];
                    
    JRow1 = matlabFunction( jacobian(RRow1, [q1; q2; q3; q4]) );
    JRow2 = matlabFunction( jacobian(RRow2, [q1; q2; q3; q4]) );
    JRow3 = matlabFunction( jacobian(RRow3, [q1; q2; q3; q4]) );
    
else
    % TODO verify that this case is not used anymore
    Rot = @(q) [-q(3)^2-q(4)^2         q(2)*q(3)-q(4)*q(1)   q(2)*q(4)+q(3)*q(1);
        q(2)*q(3)+ q(4)*q(1)   -q(2)^2- q(4)^2       q(3)*q(4)-q(2)*q(1);
        q(2)*q(4)-q(3)*q(1)    q(3)*q(4)+q(2)*q(1)   -q(2)^2-q(3)^2      ];
end
end

