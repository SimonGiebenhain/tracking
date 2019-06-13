%TODO check implay function mal ab


%% Load Data
%%%%% Manually %%%%%
%readtable('../datasets/20190124_10BirdsWeightTrials05_testdata.csv', 'ReadRowNames', true, 'HeaderLines', 1);
%% Discard rotation and separate X,Y,Z coordinates
[N,C] = size(D_unlabeled);
K =10;
%%
figure;
scatter3([X_min, X_max], [Y_min, Y_max], [Z_min, Z_max], '*')
hold on
P = cell(K,1);
for k = 1:K
    points = D_clustered{1,k};
    
    %TODO wie geht list richtig?
    if k < 5
        if size(points,1) > 0
            P{k} = plot3(points(:,1),points(:,2),points(:,3), 'o', 'MarkerSize', 10);
        else
            P{k} = plot3(0,0,0, 'o', 'MarkerSize', 10);
        end
    else
        if size(points,1) > 0
            P{k} = plot3(points(:,1),points(:,2),points(:,3), '+', 'MarkerSize', 10);
        else
            P{k} = plot3(0,0,0, '+', 'MarkerSize', 10);
        end
    end
        
end
P_unlabeled = plot3(D_unlabeled(1,1:3:C),D_unlabeled(1,2:3:C),D_unlabeled(1,3:3:C),'*','MarkerEdgeColor','green');

grid on;
hold off
axis manual
%%
for t = 2200:N
    for k = 1:K
        points = D_clustered{t,k};
        if size(points,1) > 0
            P{k}.XData = points(:,1);
            P{k}.YData = points(:,2);
            P{k}.ZData = points(:,3);
        else
            P{k}.XData = NaN;
            P{k}.YData = NaN;
            P{k}.ZData = NaN;
        end
    end
    P_unlabeled.XData = D_unlabeled(t,1:3:C);
    P_unlabeled.YData = D_unlabeled(t,2:3:C);
    P_unlabeled.ZData = D_unlabeled(t,3:3:C);
    drawnow 
    pause(0.1)
    t
end