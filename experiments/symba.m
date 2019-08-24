allPatterns = read_patterns('tracking/datasets/framework');
patterns = zeros(10,4,3);
idx = 1;
for i=1:length(allPatterns)
    patterns(idx,:,:) = allPatterns(i).pattern;
    idx = idx + 1;
end
[H, J] = getMeasurementFunction(allPatterns(1).pattern, 'brownian', 'test');

%%
fJ = @() J(5,0.25,25,0.25); % handle to function
timeJNew = timeit(fJ)

fOpt = @() test(5,0.25,25,0.25);
timeJOpt = timeit(fOpt)

%x = [10; 10; 10; 0; 0; 0; 1; 0; 0; 0;];
%fH = @() H(x); % handle to function
%timeHNew = timeit(fH)


