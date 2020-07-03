function res = Ad(S)
%AD The adjoint operator of the Lie Group, maps element from Lie group to
% a linear function on the Lie algebra.
%   This implementation of the adjoint representation is not the regular
%   one, but instead represents elements of the Lie Group on R^p, again by
%   using the isomorphism between the Lie algebra and R^p.
%   See https://hal.archives-ouvertes.fr/hal-00903252/document for more
%   details.
%
%   Arguements:
%   @S element of the Lie Group
%
%   Returns:
%   @res matrix of dimensions [6x6], [12x12] depening on the used motion
%   model. Here a constant velcity model is not implemented.
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
    error('constant Velocity model not yet implemented')
end
end

