function result = roll_avg(magnet_data,delay)

roll_avg = nan(1,length(magnet_data));

for ii = (1+delay):(length(magnet_data)-delay)
    roll_avg(ii) = mean(magnet_data(ii-delay:ii+delay));
end

result = roll_avg(~isnan(roll_avg));

end