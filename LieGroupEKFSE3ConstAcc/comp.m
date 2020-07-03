function S = comp(S1, S2)
%COMP Group composition of SE(3) x R^3 (x R^3)
%   The group composition for SE(3) is the regular matrix multiplication.
%   For this Lie Group the second and third component preserve the group
%   composition from Euclidean space (Addition). Depending on the motion
%   model, different amounts of additions have to be carried out.
%   Arguments:
%   @S1 element from Lie Group
%   @S2 element from Lie Group
%
%   Returns:
%   @S element from Lie Group


S.X = S1.X*S2.X;
S.motionModel = S1.motionModel;
if S.motionModel > 0
    S.v=S1.v+S2.v;
end
if S.motionModel == 2
    S.a=S1.a+S2.a;
end
end

