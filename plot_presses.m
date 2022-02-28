function plot_presses(normalized_data)

lever_ind = normalized_data{1};
normalized = normalized_data{2};
curr_press = 1;

figure(1)
hold on

for ii = 1:length(lever_ind)
    for jj = 1:length(lever_ind)
        press_1 = normalized(lever_ind(ii,1):lever_ind(ii,2));
        press_2 = normalized(lever_ind(jj,1):lever_ind(jj,2));
        if length(press_1) > length(press_2)
            press_2 = [press_2 ones(1, length(press_1) - length(press_2))];
        elseif length(press_2) > length(press_1)
            press_1 = [press_1 ones(1, length(press_2) - length(press_1))];
        end
        curr_covar = cov(press_1,press_2);
        r_values(length(lever_ind)+1-ii,jj) = curr_covar(1,2)/sqrt(curr_covar(1,1) * curr_covar(2,2));
%         press_1 = normalized(lever_ind(ii,1):lever_ind(ii,1)+300);
%         press_2 = normalized(lever_ind(jj,1):lever_ind(jj,1)+300);
    end
    if mod(curr_press,25) == 0
        hold on
        plot(press_1)
        plot(press_2)
    end
    curr_press = curr_press + 1;
end
end