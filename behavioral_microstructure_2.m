function behavioral_microstructure_2(subject_num, date_input,my_height,wanttosave)

file_directory = specific_file_directory_maker(subject_num, date_input);
filename = file_directory{2,2};
data = data_extractor(filename);
lever_data = data{1};
copy_lever_data = data{1};
reward_data = data{2};
lick_data = data{3};

hold on
plot(lever_data, 1:length(lever_data),'.k','MarkerSize',10)

% Plot licks before first reward
beg_licks = lick_data(lick_data < lever_data(1));
for ii = 1:length(beg_licks)
    plot([beg_licks(ii) beg_licks(ii)],[0 my_height],'-r');
end

% Plot licks
for ii = 1:length(lever_data)
    curr_lever = lever_data(ii);
    curr_lever_ind = ii;
    if ii < length(lever_data)
        temp_lick = lick_data(lick_data > curr_lever & lick_data < lever_data(ii+1));
        num_licks = length(temp_lick);
        if ~isempty(temp_lick)
            for jj = 1:num_licks
                plot([temp_lick(jj) temp_lick(jj)],[curr_lever_ind-my_height, curr_lever_ind],'-r');
            end
        end
    else
        temp_lick = lick_data(lick_data > curr_lever);
        num_licks = length(temp_lick);
        if ~isempty(temp_lick)
            for jj = 1:num_licks
                plot([temp_lick(jj) temp_lick(jj)],[curr_lever_ind-my_height, curr_lever_ind],'-r');
            end
        end
    end
end

% Plot reward lines 
for ii = 1:length(reward_data)  
    curr_seq = copy_lever_data(copy_lever_data < reward_data(ii));
    last_press = length(curr_seq(1:end-4)) + 4;
    plot([reward_data(ii), reward_data(ii)], [0, last_press],'--k')
end

my_title = sprintf('Behavioral Microstructure: #%d on %s',subject_num,...
    date_input(6:end));
title(my_title);
xlabel('t(s)')
ylabel('Number of presses')

if wanttosave
    path='C:\Users\emmag\OneDrive - Vanderbilt\Documents\Calipari Lab =)\Data\Graphs\dynrespwind nov21';
    filename = sprintf('Behavior_%d_%s.fig', subject_num, date_input(6:end));
    saveas(gcf,fullfile(path,filename));
end

hold off
fclose('all');

end