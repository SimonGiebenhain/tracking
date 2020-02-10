pattern = [1 1 1; 1 0 0; -1 1 0; 0 0 -1];
pattern = [pattern ones(4,1)];
theta = 0.3;
XInit = [1 0 0 0; 0 cos(theta) -sin(theta) 0; 0 sin(theta) cos(theta) 0; 0 0 0 1];
vInit = [0; 0; 0];
aInit = [0; 0; 0];
muInit.X = XInit;
muInit.v = vInit;
muInit.a = aInit;
% PInit = diag([1;1;1;10;10;10;5;5;5;3;3;3]);
% R = diag([0.5;0.5;0.5;2;2;2;5;5;5;5;5;5]);
% Q = diag(repmat([5; 5; 5; 5], 4, 1));
PInit = 7*diag([0.3 0.3 0.3 0.5 0.5 0.5 0.2 0.2 0.2 0.2 0.2 0.2]);
R = 2*diag([0.5 0.5 0.5 1 1 1 1 1 1 1 1 1]);
Q = 1/2 * eye(16);
%measurement function, h
h = @(S) [measFunc(S,pattern(1,:)'); measFunc(S,pattern(2,:)'); measFunc(S,pattern(3,:)'); measFunc(S,pattern(4,:)')];

%Jacobian of h, H
jac = @(m, mu) [
    HLinSE3AC(m(1,1), m(1,2), m(1,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3));
    HLinSE3AC(m(2,1), m(2,2), m(2,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3));
    HLinSE3AC(m(3,1), m(3,2), m(3,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3));
    HLinSE3AC(m(4,1), m(4,2), m(4,3), mu(1,1), mu(1,2), mu(1,3), mu(2,1), mu(2,2), mu(2,3), mu(3,1), mu(3,2), mu(3,3))
    ];

H = @(S) jac(pattern, S.X);


mu = muInit;
P = PInit;
lossRot = zeros(1000,1);
lossTrans = zeros(1000,1);


tic

for i=1:1000
    %theta = i*pi/100;
    x_off = sin(i/10);
    y_off = i / 30;
    z_off = (-cos(i/10));
    q = [sin(i/100) cos(i/100)*2 (i-500)/100 -sin(i/100)^2];
    q = q/sqrt(sum(q.^2));
    %rot_mat = [1 0 0 x_off; 0 cos(theta) -sin(theta) y_off; 0 sin(theta) cos(theta) z_off; 0 0 0 1];
    rot_mat = quat2rotm(q);
    rot_mat = [rot_mat [x_off;y_off;z_off]];
    rot_mat = [rot_mat; [0 0 0 1]];
    dets = (rot_mat * pattern');
    dets = reshape(dets, 16,1) + normrnd(0, 0.005, 16, 1);
    
    [mu, P] = predict(mu, P, R);
    [mu, P] = update(dets, mu, P, H, h, Q);
    %mu
    mu.X - rot_mat
    lossRot(i) = mean(abs(mu.X(1:3,1:3)-rot_mat(1:3,1:3)), 'all');
    lossTrans(i) = mean(abs(mu.X(1:3,4)-rot_mat(1:3, 4)), 'all');
end
toc
figure;
plot(lossRot) 
hold on;
plot(lossTrans)
%figure;
%plot((1:1000)*pi/300)


function [mu, P] = predict(mu, P, R)
F = JacOfFonSE3CA(mu);
mu = comp(mu, expSE3ACvec(stateTrans(mu)));
P = F*P*F' + R;
end

function [mu, P] = update(z, mu, P, H, h, Q)
Hk = H(mu);
K = P*Hk'/(Hk*P*Hk'+Q);
m = K*(z-h(mu));
mu = comp(mu, expSE3ACvec(m));
P = (eye(12)- K*Hk)*P;
end

