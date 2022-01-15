function lever_ind = detect_lever_ind(roll_matrix,threshold)

% Initialization-----------------------------------------------------------
magnet_roll = roll_matrix{1};
time_roll = roll_matrix{2};
delay = roll_matrix{3};

% figure(1)
hold on
xlabel('Time (s)')

magnet_off = median(magnet_roll(time_roll < time_roll(1) + 1000));
on = abs(magnet_roll - magnet_off) > 230;
magnet_on = magnet_roll(on);
time_on = time_roll(on);
orig_baseline = median(magnet_on(1:1000));
start_ind = length(time_roll(time_roll <= time_on(1)));

% plot([time_roll(start_ind)/1000, time_roll(start_ind + 1000)/1000],...
%     [orig_baseline, orig_baseline],'Color','k','LineWidth',4)
% plot(time_roll/1000,magnet_roll,'Color','#dfe2e4')
% plot(time_roll(on)/1000,magnet_on,'Color','b')


% Calculate Baselines------------------------------------------------------

% Flag places to potentially calculate baseline
temp = magnet_off + 245;
% plot([time_roll(1)/1000,time_roll(end)/1000],[temp, temp])
baseline_flags = [];

for ii = 2:length(time_roll(time_roll <= time_on(end)))-1
    cond_1 = magnet_roll(ii + 1) < temp && magnet_roll(ii) > temp;
    cond_2 = magnet_roll(ii - 1) < temp && magnet_roll(ii) > temp;
    if cond_1
        baseline_flags = [baseline_flags ii];
%         plot(time_roll(ii)/1000,magnet_roll(ii),'MarkerSize',10,'Marker','o','Color','r')
    elseif cond_2
        baseline_flags = [baseline_flags ii];
%         plot(time_roll(ii)/1000,magnet_roll(ii),'MarkerSize',10,'Marker','o','Color','g')
    end
end

% Actual calculation
baselines = [];
baseline_ind = [];
row = 0;
baseline_flags = baseline_flags(1:length(baseline_flags)-mod(length(baseline_flags),2));
for ii = 1:length(baseline_flags)
    if mod(ii,2) == 1
        baseline_start = baseline_flags(ii);
        if baseline_flags(ii+1) - baseline_flags(ii) > 5000
            baseline_end = baseline_start + 1000;
            row = row + 1;
        elseif baseline_flags(ii+1) - baseline_flags(ii) > 1000
            baseline_end = baseline_flags(ii + 1);
            row = row + 1;
        end   
    else
        if baseline_flags(ii) - baseline_flags(ii-1) > 5000
            baseline_start = baseline_flags(ii) - 1000;
            baseline_end = baseline_flags(ii);
            row = row + 1;
        end
    end
    if mod(ii,2) == 1 && baseline_flags(ii+1) - baseline_flags(ii) < 1000
%         plot([time_roll(baseline_flags(ii-1))/1000, time_roll(baseline_flags(ii + 1))/1000], ...
%          [baselines(row), baselines(row)], 'LineWidth',1,'Color','r', ...
%          'LineStyle','--')
    elseif mod(ii,2) == 1 || (mod(ii,2) == 0 && baseline_flags(ii) - baseline_flags(ii-1) > 5000)
        baseline_chunk = magnet_roll(baseline_start:baseline_end);
        baseline_ind(row,1) = baseline_start;
        baselines = [baselines; median(baseline_chunk)];
%         plot([time_roll(baseline_start)/1000, time_roll(baseline_end)/1000], ...
%         [median(baseline_chunk), median(baseline_chunk)], 'LineWidth',2,'Color','r')
    end
end

% plot(time_roll(baseline_ind)/1000, 760 * ones(1,length(baselines)), '->', ...
%     'LineWidth',3)


% Normalization------------------------------------------------------------
% figure(2)
hold on
xlabel('Time (s)')
title('Normalized Data')

normalized = zeros(1,length(magnet_roll));

for ii = 1:length(baseline_ind)-1
    mask = baseline_ind(ii):baseline_ind(ii+1);
    curr_chunk = magnet_roll(mask)/baselines(ii);
    normalized(mask) = curr_chunk;
end


normalized(1:baseline_ind(1)) = magnet_roll(1:baseline_ind(1))/baselines(1);
normalized(baseline_ind(end):end) = magnet_roll(baseline_ind(end,1):end)/baselines(end);

% plot(time_roll/1000,normalized,'Color','#dfe2e4')
% plot([time_roll(1)/1000,time_roll(end)/1000],[1, 1],'r')
% plot([time_roll(1)/1000,time_roll(end)/1000],[0.9, 0.9],'r')

% Find indices
% plot([time_roll(1)/1000,time_roll(end)/1000],[threshold, threshold])
press_num = 1;
lever_ind = [];
%  mean(normalized((search_ind + 1):search_ind + 5)) > normalized(search_ind)
% mean(normalized(search_ind -5:search_ind-1)) > normalized(search_ind)
for ii = (baseline_flags(1) + 1):length(time_roll(time_roll <= time_on(end)))
    cond_1 = normalized(ii + 1) < threshold && normalized(ii) > threshold;
    cond_2 = normalized(ii - 1) < threshold && normalized(ii) > threshold;
    if cond_1
        searching = true;
        search_ind = ii;
        while searching
            if normalized(search_ind - 1) > normalized(search_ind)
                search_ind = search_ind - 1;
            elseif sum(normalized(search_ind - 50:search_ind-1) > normalized(search_ind))
                search_ind = search_ind - 1;
            else
                searching = false;
            end
            if normalized(search_ind-1) > 1
                searching = false;
            end
        end
        lever_ind(press_num, 1) = search_ind;
        % plot(time_roll(search_ind)/1000,normalized(search_ind),'MarkerSize',10,'Marker','o','Color','g')
    elseif cond_2
        searching = true;
        search_ind = ii;
        while searching
            if normalized(search_ind + 1) > normalized(search_ind)
                search_ind = search_ind + 1;
            elseif sum(normalized((search_ind+1):search_ind + 50) > normalized(search_ind))
                search_ind = search_ind + 1;
            else
                searching = false;
            end
            if normalized(search_ind+1) > 1
                searching = false;
            end
        end
        lever_ind(press_num, 2) = search_ind;
        press_num = press_num + 1;
        %plot(time_roll(search_ind)/1000,normalized(search_ind),'MarkerSize',10,'Marker','*','Color','r')
    end
end
end
% plot(time_roll(unique(baselines))/1000,ones(1,length(unique(baseline_ind))),...
%     'Marker','*', 'MarkerSize',10)

%{
Useful plots
Threshold:
plot([time_roll(1)/1000,time_roll(end)/1000],[threshold, threshold])

Start of magnet on:
plot(time_roll(start_ind)/1000,magnet_roll(start_ind),'MarkerSize',10,'Marker','*')
%}
