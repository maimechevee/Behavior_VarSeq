function optimal_threshold(magnet_data,lever_data, magnet_off, threshold_ON)

magnet_roll = magnet_data{1};
time_roll = magnet_data{2};
delay = magnet_data{3};

on = abs(magnet_roll - magnet_off) > threshold_ON;
magnet_on = magnet_roll(on);
time_on = time_roll(on);

num_trials = 50; %number of trials to use in optimization
min_threshold = 0.8; %lower bound of test values
max_threshold = 0.9; %upper bound of test values
increment = (max_threshold - min_threshold)/num_trials;
thresholds = min_threshold:increment:max_threshold;
result = zeros(1,num_trials+1);
for ii = 1:num_trials+1
    threshold = thresholds(ii);
    num_presses = length(detect_lever_ind(magnet_data, threshold,0,lever_data, magnet_off,magnet_on, time_on, threshold_ON));
    result(ii) = num_presses;
end

plot(thresholds,result,'-o')
title('Correct # = 255')
xlabel('Threshold')
ylabel('Num Presses')

end