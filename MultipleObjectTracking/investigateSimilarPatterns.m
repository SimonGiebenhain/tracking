deltas = linspace(0.1, 3, 50);
avgNumSimPatterns = zeros(length(deltas), 1);
for i = 1:length(deltas)
    d = deltas(i);
    simPairs = getSimilarPatterns(patterns, d);
    avgNumSimPatterns(i) = (size(simPairs,1)*2)/size(patterns,1);
end

figure;
plot(deltas, avgNumSimPatterns)
