function optimal_threshold(roll_matrix)
magnet_roll = roll_matrix{1};
time_roll = roll_matrix{2};
delay = roll_matrix{3};
magnet_off = median(magnet_roll(time_roll < time_roll(1) + 1000));
on = abs(magnet_roll - magnet_off) > 230;
magnet_on = magnet_roll(on);
time_on = time_roll(on);

num_trials = 100;
min_threshold = 0.79;
max_threshold = 0.97;
increment = (max_threshold - min_threshold)/num_trials;
thresholds = min_threshold:increment:max_threshold;
result = zeros(1,num_trials+1);
for ii = 1:num_trials+1
    threshold = thresholds(ii);
    num_presses = length(detect_lever_ind(roll_matrix, threshold));
    result(ii) = num_presses;
end
plot([thresholds(1),thresholds(end)],[255,255])

plot(thresholds,result,'-o')
title('Correct # = 255')
xlabel('Threshold')
ylabel('Num Presses')

end