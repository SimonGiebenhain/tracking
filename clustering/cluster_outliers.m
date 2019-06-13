[T,~] = size(D_unlabeled);
D_clustered = cell(T,10);
for t = 1:T
   frame = D_unlabeled(t,:);
   if length(frame(~isnan(frame))) < 6
       continue
   end
   frame = frame(~isnan(frame));
   frame = reshape(frame, [],1);
   N = size(frame,1);
   frame3D = cat(2, frame(1:3:N), frame(2:3:N), frame(3:3:N));
   d = pdist(frame3D);
   dendro = linkage(d);
   inconsistent(dendro);
   % figure
   % dendrogram(dendro)
   clusters = cluster(dendro, 'cutoff', 0.1*1e03, 'Criterion', 'distance');
   num_clusters = max(clusters);
   % TODO wie geht eine List, oder cell array oder was ist beste Lösung in
   % MATLAB?
   for c = 1:num_clusters
       D_clustered{t,c} = frame3D(clusters == c,:);
   end
   
   %figure; hold on; grid on;
   %for c = 1:num_clusters
   %   plot3(D_clustered{t,c}(:,1), D_clustered{t,c}(:,2), D_clustered{t,c}(:,3), 'o') 
   %end
   %hold off;
   
   % Checke in Vergangenheit und Zukunft, i.e. temporal vicinity, ob Vogel
   % in Nähe
   
   % Wenn nicht verwerfe cluster, aber Zeige ihn deutlich sichtbar an
   
   % Wenn ja, mache cluster zu detection von geleichem vogel
end