deltas = linspace(0.1, 3, 50);
avgNumSimPatterns = zeros(length(deltas), 1);
for i = 1:length(deltas)
    d = deltas(i);
    simPairs = getSimilarPatterns(patterns, d);
    for j = 1:size(simPairs,1)
       l = simPairs(j,1);
       r = simPairs(j,2);
       if r < l
          simPairs(j,1) = r;
          simPairs(j,2) = l;
       end
    end
    simPairs = unique(simPairs, 'rows');
    avgNumSimPatterns(i) = (size(simPairs,1)*2)/size(patterns,1);
end

figure;
plot(deltas, avgNumSimPatterns)
