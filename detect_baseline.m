% Returns start index of median value of the first 1000 datapoints after the magnet
% turns on 

function [beg_ind, baseline, magnet_off_mean, Threshold] = detect_baseline(magnet_data, delay)

magnet_roll = magnet_data{1};
time_data = magnet_data{2};
roll_time = time_data((1+delay):(length(magnet_roll)-delay));

% Grab data from first second of file when magnet is still off
magnet_off_mean = mean(magnet_roll(roll_time < roll_time(1) + 1000)); %get mean of baseline
magnet_off_std = std(magnet_roll(roll_time < roll_time(1) + 1000)); %get std of baseline
Threshold=20*magnet_off_std; %threshold 

not_found = true;
beg_ind = 1;
ii = 1;

while not_found %loop through data until you deflect from baseline> break while loop
    if (abs(magnet_roll(ii) - magnet_off_mean) > Threshold) 
        beg_ind = ii;
        not_found = false;
    end
    ii = ii + 1;
end

% Grab baseline of first second of when magnet turns off 
baseline = mean(magnet_roll(beg_ind:beg_ind + 1000));

end