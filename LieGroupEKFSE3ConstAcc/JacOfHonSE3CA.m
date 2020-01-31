syms e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 e12 real;
syms m1 m2 m3 real;
syms mu1 mu2 mu3 mu4 mu5 mu6 mu7 mu8 mu9 mu10 mu11 mu12 real;
syms v1 v2 v3 a1 a2 a3 real;

mu = [mu1 mu2 mu3 mu4; mu5 mu6 mu7 mu8; mu9 mu10 mu11 mu12; 0 0 0 1];
v = [v1; v2; v3];
a = [a1; a2; a3];
m = [m1; m2; m3; 1];
e = [e1; e2; e3; e4; e5; e6; e7; e8; e9; e10; e11; e12];
%%
S.X = mu;
S.v = v;
S.a = a;

J = jacobian(measFunc(comp(S, expSE3CAvec(e)),m), [e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 e12]);

%%
syms t real;
J = subs(J, [e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 e12], [t t t t t t t t t t t t]);
%%
J = limit(J, t, 0)

%%
j = matlabFunction(J, 'File', 'HLinSE3AC')

jac = @(m, mu) [
    HLinSE3AC(m(1,1), m(1,2), m(1,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3));
    HLinSE3AC(m(2,1), m(2,2), m(2,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3));
    HLinSE3AC(m(3,1), m(3,2), m(3,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3));
    HLinSE3AC(m(4,1), m(4,2), m(4,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3))
    ];



