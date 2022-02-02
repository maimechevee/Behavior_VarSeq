% Returns start index of median value of the first 1000 datapoints after the magnet
% turns on 

function [beg_ind, baseline, magnet_off] = detect_baseline(magnet_matrix, magnet_roll, lever_data, delay)

magnet_data = magnet_matrix(:,1);
time_data = magnet_matrix(:,3);
roll_time = time_data((1+delay):(length(magnet_data)-delay));

% Grab data from first second of file when magnet is still off
magnet_off = mean(magnet_roll(roll_time < roll_time(1) + 1000));

not_found = true;
beg_ind = 1;
ii = 1;

while not_found
    if abs(magnet_roll(ii) - magnet_off) > 250
        beg_ind = ii;
        not_found = false;
    end
    ii = ii + 1;
end

% Grab baseline of first second of when magnet turns off 
baseline = mean(magnet_roll(beg_ind:beg_ind + 1000));

end