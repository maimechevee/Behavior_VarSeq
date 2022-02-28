function plot_presses(normalized_data, x)

lever_ind = normalized_data{1};
normalized = normalized_data{2};
curr_press = 1;

figure(1)
hold on

for ii = x:x+9
    plot(normalized(lever_ind(ii,1):lever_ind(ii,2)))
end

title('Overlay of 10 Presses')
xlabel('time (ms)')
ylabel('hall sensor')
end