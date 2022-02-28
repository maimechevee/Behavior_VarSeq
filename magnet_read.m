function magnet_data = magnet_read(file,delay)

% Read CSV file into magnet_data - a cell array where first item is the
%  rolling average of the sensor data, the second item is the cropped
% time values corresponding to the rolling average, and the delay is 
% the delay used for rolling average (rolling average taken from x-delay 
% to x + delay

magnet_matrix = readmatrix(file);
magnet_vector = magnet_matrix(:,1);
time_vector = magnet_matrix(:,3);

magnet_data = {};
roll_data = NaN(1,length(magnet_vector));

for ii = (1+delay):(length(magnet_vector)-delay)
    roll_data(ii) = mean(magnet_vector(ii-delay:ii+delay));
end

magnet_data{1} = roll_data(~isnan(roll_data));
magnet_data{2} = time_vector((1+delay):(length(magnet_vector)-delay));
magnet_data{3} = delay;

end