function r_values = single_day_corr(roll_matrix, data)

lever_ind = data{1};
normalized = data{2};

magnet_roll = roll_matrix{1};
time_roll = roll_matrix{2};
delay = roll_matrix{3};

r_values = zeros(length(lever_ind),length(lever_ind));


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
    end
    if mod(ii,10) == 0
        figure(ii)
        hold on
        plot(press_1)
        plot(press_2)
        hold off
    end
end

figure(1)
clrLim = [-1, 1];
imagesc(r_values)
set(gca, 'YTick', [256:50:1])
colormap(gca,'cool');
colorbar();
caxis(clrLim);
axis equal
axis tight
my_title = sprintf("#4220, Last Day of Training");
title(my_title)
xlabel('Press #')
ylabel('Press #')
caxis([0, 1])

end