function result = roll_avg(magnet_matrix,delay)

magnet_data = magnet_matrix(:,1);
time_data = magnet_matrix(:,3);

result = {};
roll_data = nan(1,length(magnet_data));

for ii = (1+delay):(length(magnet_data)-delay)
    roll_data(ii) = mean(magnet_data(ii-delay:ii+delay));
end

result{1} = roll_data(~isnan(roll_data));
result{2} = time_data((1+delay):(length(magnet_data)-delay));
result{3} = delay;

end