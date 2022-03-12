function plot_presses_NormIncluded(normalized_data, x)

lever_ind = normalized_data{1};
normalized = normalized_data{2};
curr_press = 1;

figure(1)
hold on

for ii = x:x+19
    Norm=normalized(lever_ind(ii,1):lever_ind(ii,2))/mean(normalized(lever_ind(ii,1)-50:lever_ind(ii,1)));
    Adjusted=Norm-Norm(1);
    plot(Adjusted)
end

title('Overlay of 10 Presses')
xlabel('time (ms)')
ylabel('hall sensor')
end