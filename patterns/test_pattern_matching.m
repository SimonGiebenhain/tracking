marker1 = [1 1; 0 0; 0 -2; -2 0];

marker2 = [1 1; 0 0; 0 -2; -2 0] + 0.5*randn(4,2);

assignemnt = match_patterns(marker1, marker2);