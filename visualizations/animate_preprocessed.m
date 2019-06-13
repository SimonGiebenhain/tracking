%TODO check implay function mal ab


%% Load Data
%%%%% Manually %%%%%
%readtable('../datasets/20190124_10BirdsWeightTrials05_testdata.csv', 'ReadRowNames', true, 'HeaderLines', 1);
%% Discard rotation and separate X,Y,Z coordinates
[N,C] = size(D_labeled);
K = C / 7;
X = zeros(N, K);
Y = zeros(N, K);
Z = zeros(N, K);
column_idx = 1;
for c = 1:70
    if mod(c,7) == 5
        X(:,column_idx) = D_labeled(:,c); 
    elseif mod(c,7) == 6
        Y(:,column_idx) = D_labeled(:,c); 
    elseif mod(c,7) == 0
        Z(:,column_idx) = D_labeled(:,c); 
        column_idx = column_idx + 1;
    end
end

% Get extreme coordinates
X_min = min(X, [], 'all');
X_max = max(X, [], 'all');
Y_min = min(Y, [], 'all');
Y_max = max(Y, [], 'all');
Z_min = min(Z, [], 'all');
Z_max = max(Z, [], 'all');
%%

scatter3([X_min, X_max], [Y_min, Y_max], [Z_min, Z_max], '*')
hold on
P = cell(K,1);
for k = 1:K
    %TODO wie geht list richtig?
    if k < 5
        P{k} = plot3(X(1,k),Y(1,k),Z(1,k), 'o', 'MarkerSize', 10);
    else
        P{k} = plot3(X(1,k),Y(1,k),Z(1,k), '+', 'MarkerSize', 10);
    end
        
end
grid on;
hold off
axis manual
%%
for t = 2:N
    for k = 1:K
        P{k}.XData = X(t,k);
        P{k}.YData = Y(t,k);
        P{k}.ZData = Z(t,k);
    end
    drawnow 
    %pause(0.01)
    t
end