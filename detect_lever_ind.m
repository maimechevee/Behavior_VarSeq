% Return beg and end indices of each lever

%{
delay = 15;
magnet_matrix_4220 = readmatrix('HallSensor_20211215_4220.CSV');
magnet_roll = roll_avg(magnet_matrix_4220(:,1),delay);
%}

function lever_ind = detect_lever_ind(magnet_matrix, magnet_roll, lever_data, delay)
magnet_data = magnet_matrix(:,1);
time_data = magnet_matrix(:,3);
[beg_ind, baseline, magnet_off] =  detect_baseline(magnet_matrix, magnet_roll, lever_data, delay);
roll_time = time_data((1+delay):(length(magnet_data)-delay));

%Grab rough outline of the lever presses
magnet_temp = magnet_roll(beg_ind:end); % cutoff beg. portion when magnet off
magnet_temp1 = magnet_temp(abs(magnet_off-magnet_temp)> 50); %cutoff end portion
magnet_temp2 = magnet_temp1(abs(baseline-magnet_temp1)> 25); %cutoff baseline

time_temp = roll_time(beg_ind:end);
time_temp = time_temp(abs(magnet_off-magnet_temp)> 50);
time_temp = time_temp(abs(baseline-magnet_temp1)> 25);

%Find beg and end indices

magnet_slice = magnet_temp2(abs(baseline-magnet_temp2)< 100 & abs(baseline-magnet_temp2)> 90);
time_slice = time_temp(abs(baseline-magnet_temp2)< 100 & abs(baseline-magnet_temp2)> 90);

prev_type = '';
press_num = 1;
lever_ind = [];

figure(1)
hold on

for ii = 1:length(magnet_slice)-1
    curr_time = time_slice(ii);
    curr_pt = magnet_slice(ii);
    roll_ind = length(roll_time(roll_time <= curr_time));
    if abs(magnet_roll(roll_ind+1) - baseline) > abs(magnet_roll(roll_ind) - baseline)
        curr_type = 'press';
    elseif abs(magnet_roll(roll_ind+1) - baseline) < abs(magnet_roll(roll_ind) - baseline)
        curr_type = 'release';
    end
    if ~strcmp(curr_type, prev_type) 
        search_ind = roll_ind;
        if strcmp(curr_type, 'press')
            while abs(magnet_roll(search_ind)-baseline) > 10 &&...
                abs(magnet_roll(search_ind-10) - baseline) < abs(magnet_roll(search_ind) - baseline)
                search_ind = search_ind - 1;
            end
            lever_ind(press_num, 1) = search_ind;
            plot(roll_time(search_ind)/1000,magnet_roll(search_ind),'Color','red',...
                'MarkerSize',10,'Marker','*')
        elseif strcmp(curr_type, 'release')
            while abs(magnet_roll(search_ind)-baseline) > 10 &&...
                    abs(magnet_roll(search_ind+10) - baseline) < abs(magnet_roll(search_ind) - baseline)
                search_ind = search_ind + 1;
            end
            lever_ind(press_num, 2) = search_ind;
            plot(roll_time(search_ind)/1000,magnet_roll(search_ind),'Color','blue',...
                'MarkerSize',20,'Marker','o')
            press_num = press_num + 1;
        end
    end
    prev_type = curr_type;
end

%Plot
figure(1)
hold on
% plot(time_data/1000,magnet_data,'LineWidth',1,'Color','#808080');
plot(roll_time/1000,magnet_roll,'LineWidth',1,'Color','#808080');
plot(roll_time(beg_ind)/1000,magnet_data(beg_ind),'Marker','o','Color','red','MarkerSize',10)
plot(lever_data + (roll_time(beg_ind))/1000, magnet_off * ones(1, length(lever_data)), Marker = 'o')
plot(time_temp/1000,magnet_temp2,'Linewidth',2,'Color','r')
plot(time_slice/1000,magnet_slice,'Linewidth',3,'Color','b')
end