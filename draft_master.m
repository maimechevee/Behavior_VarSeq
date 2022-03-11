file='H:\Maxime\Hall Sensor Data\20220131\HallSensor_20220131_4229.csv'
delay=20
magnet_data = magnet_read(file,delay)
filename='G:\Behavior study Dec2021\All medpc together\2022-01-31_16h04m_Subject 4229.txt'
lever_data = extract_lever_data(filename)

%Look at data
plot(magnet_data{1})

%detect start of session
[beg_ind, orig_baseline, magnet_off, threshold_ON] = detect_baseline(magnet_data, delay);

optimal_threshold(magnet_data,lever_data)