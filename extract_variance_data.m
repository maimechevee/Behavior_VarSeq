function variance_data = extract_variance_data(filename)

    fid = fopen(filename, 'rt');
    
    if fid < 0
        fprintf('Error opening file.\n');
        return;
    end

    last_row = 0;
    first_row = 0;
    line_num = 1;
    current_line = fgetl(fid);
    found_last_row = false;

    while found_last_row == false
        if sum(size(current_line)) == 0 
            line_num = line_num + 1;
            current_line = fgetl(fid);
        elseif current_line(1) == 'F' && current_line(2) == ':'
            line_num = line_num + 1;
            first_row = line_num;
            current_line = fgetl(fid);
        elseif current_line(1) == 'G' && current_line(2) == ':'
            line_num = line_num + 1;
            last_row = line_num;
            break;
        else 
            line_num = line_num + 1;
            current_line = fgetl(fid);
        end
    end

    opts = delimitedTextImportOptions('Delimiter',{' ',':'},...
        'ConsecutiveDelimitersRule','join','LeadingDelimItersRule','ignore');
    opts.DataLines = [first_row+1 last_row-1]; % sets line range to scan
    opts.EmptyLineRule = 'skip';
    messy_matrix = readcell(filename,opts);
    
    size_raw_matrix = size(messy_matrix);
    size_raw_matrix_rows = size_raw_matrix(1);
    size_raw_matrix_cols = size_raw_matrix(2);
    variance_data = [];

    for ii = 1:size_raw_matrix_rows
        for jj = 2:size_raw_matrix_cols
            if isnumeric(messy_matrix{ii,jj})
                curr_data = messy_matrix{ii,jj};
                variance_data = [variance_data; curr_data];
            end
        end
    end
end
