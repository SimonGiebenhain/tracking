function Rot = quatToMat()
%QUATTOMAT Construct rotation matrix from quaternion
%   Using the formula from: https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
%   this function constructs a function handle which outputs a 3x3 rotation matrix from a supplied quaternion with the
%   same rotation properties, i.e. Rot(q)*x = q*x*q^(-1)
%   The function normalizes the quaternion before applying the formula.


% if ~exist('normalize','var') && ~exist('q', 'var')
%     normalize = 1;
% end
% if normalize

Rot = @(q) eye(3) + 2/sum(q.^2) * ...
                    [-q(3)^2-q(4)^2         q(2)*q(3)-q(4)*q(1)  q(2)*q(4)+q(3)*q(1);
                      q(2)*q(3)+ q(4)*q(1) -q(2)^2- q(4)^2       q(3)*q(4)-q(2)*q(1);
                      q(2)*q(4)-q(3)*q(1)   q(3)*q(4)+q(2)*q(1) -q(2)^2-q(3)^2      ;];

% else
%     Rot = @(q) [-q(3)^2-q(4)^2         q(2)*q(3)-q(4)*q(1)  q(2)*q(4)+q(3)*q(1);
%                  q(2)*q(3)+ q(4)*q(1) -q(2)^2- q(4)^2       q(3)*q(4)-q(2)*q(1);
%                  q(2)*q(4)-q(3)*q(1)   q(3)*q(4)+q(2)*q(1) -q(2)^2-q(3)^2      ;];
% end
end

