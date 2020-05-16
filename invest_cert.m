load('c_wrong_complete.mat')
c_wrong = certainties;
avg_wrong_c = mean(certainties, 'all', 'omitnan');
load('c_correct_complete.mat')
c_correct = certainties;
avg_correct_c = mean(certainties, 'all', 'omitnan');


colors = distinguishable_colors(11);
% for i=1:3
%     plot(certainties(i,:), 'color', colors(i,:))
% end
for i=1:11
    figure; hold on;
    plot(c_wrong(i,:)-avg_wrong_c);
    plot(c_correct(i,:)-avg_correct_c);
    hold off;
end

