function reorganized = reorganize(r_vals_array, new_ind)

% Draw heat map and reorganize order based on clustering

% r_vals_array is taken from single_day_corr.m
% new_ind is taken from the python script from the dendrogram

presses = r_vals_array{2};
new_ind = new_ind + 1; % Python starts at 0
max_press_length = 1;

% Find max press length
for ii = 1:length(presses)
    curr_length = length(presses{ii});
    if curr_length > max_press_length
        max_press_length = curr_length;
    end
end

reorganized = ones(length(presses), max_press_length);

% Reorganize rows based on new indices
for ii = 1:length(new_ind)
    % reorganized(ii,1:length(presses{new_ind(ii)})) = presses{new_ind(ii)};
    reorganized(ii,1:200) = presses{new_ind(ii)}(1:200);
end

imagesc(reorganized(:,1:200))
colormap parula
end