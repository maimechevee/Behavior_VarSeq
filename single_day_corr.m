function r_vals_array = single_day_corr(normalized_data)

%{
normalized_data - cell array with first item as start and end indices
    of lever presses, second item is normalized vector of the entire
    session
r_vals_array - cell array with first item as matrix of r values and second item
    as another cell array with all of the press vectors (made by slicing into normalized 
    vector using start and end indices) 
%}

lever_ind = normalized_data{1};
normalized = normalized_data{2};

r_values = zeros(length(lever_ind),length(lever_ind));
presses = {};
curr_press = 1;

for ii = 1:length(lever_ind)
    for jj = 1:length(lever_ind)
%         press_1 = normalized(lever_ind(ii,1):lever_ind(ii,2));
%         press_2 = normalized(lever_ind(jj,1):lever_ind(jj,2));
%         if length(press_1) > length(press_2)
%             press_2 = [press_2 ones(1, length(press_1) - length(press_2))];
%         elseif length(press_2) > length(press_1)
%             press_1 = [press_1 ones(1, length(press_2) - length(press_1))];
%         end
%         curr_covar = cov(press_1,press_2);
%         r_values(length(lever_ind)+1-ii,jj) = curr_covar(1,2)/sqrt(curr_covar(1,1) * curr_covar(2,2));
        press_1 = normalized(lever_ind(ii,1):lever_ind(ii,1)+300);
        press_2 = normalized(lever_ind(jj,1):lever_ind(jj,1)+300);
        curr_covar = cov(press_1,press_2);
        % r_values(length(lever_ind)+1-ii,jj) = curr_covar(1,2)/sqrt(curr_covar(1,1) * curr_covar(2,2));
        r_values(ii, jj) = curr_covar(1,2)/sqrt(curr_covar(1,1) * curr_covar(2,2));
    end
%     if mod(curr_press,75) == 0
%         figure(ii)
%         hold on
%         plot(press_1)
%         plot(press_2)
%      end
    presses{curr_press} = press_1;
    curr_press = curr_press + 1;
end

figure(1)
clrLim = [-1, 1];
imagesc(r_values)
colormap(gca,'cool');
colorbar();
caxis(clrLim);
axis equal
axis tight
my_title = sprintf("#4220, Last Day of Training");
title(my_title)
xlabel('Press #')
ylabel('Press #')
caxis([0.5, 1])

r_vals_array = {r_values, presses};
save("r_vals_array.mat","r_vals_array")

end