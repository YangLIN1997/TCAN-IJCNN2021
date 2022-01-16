from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import h5py
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def prep_data(data, name='train'):
    input_size = window_size-stride_size
    time_len = data.shape[0]
    total_windows = (time_len-input_size) // stride_size
    print("windows pre: ", total_windows)
    x_input = np.zeros((total_windows, window_size, data.shape[1]-4), dtype='float32')
    label = np.zeros((total_windows, window_size), dtype='float32')
    for i in range(total_windows):
        window_start = stride_size * i
        forecast_start = window_start + input_size
        window_end = window_start + window_size
        x_input[i, 1:, 0] = data[window_start:window_end - 1, 0]
        x_input[i, :input_size, 1:] = data[window_start:forecast_start, 1:-4]
        x_input[i, input_size:, 1:5] = data[forecast_start:window_end, -4:]
        x_input[i, input_size:, 5:] = data[forecast_start:window_end, 5:-4]
        label[i, :] = data[window_start:window_end, 0]


    prefix = os.path.join(save_path, name+'_')

    np.save(prefix+'data_'+save_name, x_input)
    print(prefix+'data_'+save_name, x_input.shape)
    # if name == 'test':
    np.save(prefix+'mean_'+save_name, data_mean)
    np.save(prefix+'scale_'+save_name, data_scale)
    np.save(prefix+'label_'+save_name, label)

def prep_data_search(data, name='train'):
    input_size = window_size-stride_size
    time_len = data.shape[0]
    total_windows = (time_len-input_size) // stride_size
    print("windows pre: ", total_windows)
    # if train: windows_per_series -= (stride_size-1) // stride_size
    x_input = np.zeros((total_windows, window_size, data.shape[1]-4), dtype='float32')
    label = np.zeros((total_windows, window_size), dtype='float32')
    for i in range(total_windows):
        window_start = stride_size * i
        forecast_start = window_start + input_size
        window_end = window_start + window_size
        x_input[i, 1:, 0] = data[window_start:window_end - 1, 0]
        x_input[i, :input_size, 1:] = data[window_start:forecast_start, 1:-4]
        x_input[i, input_size:, 1:5] = data[forecast_start:window_end, -4:]
        x_input[i, input_size:, 5:] = data[forecast_start:window_end, 5:-4]
        label[i, :] = data[window_start:window_end, 0]


    prefix = os.path.join(save_path, name+'_')
    np.save(prefix+'data_'+'search_'+save_name, x_input)
    print(prefix+'data_'+'search_'+save_name, x_input.shape)
    np.save(prefix+'scale_'+'search_'+save_name, data_scale)
    if name == 'test':
        np.save(prefix+'mean_'+'search_'+save_name, data_mean)
        np.save(prefix+'label_'+'search_'+save_name, label)
    np.save(prefix+'label_'+'search_'+save_name, label)

def visualize(data, day_start,day_num=1):
    x = np.arange(stride_size*day_num)
    f = plt.figure()
    plt.plot(x, data[day_start*stride_size:day_start*stride_size+stride_size*day_num,0], color='b')
    f.savefig("visual.png")
    plt.close()

if __name__ == '__main__':

    global save_path
    name = 'LD2011_2014.txt'
    save_name = 'Sanyo'
    window_size = 40
    stride_size = 20
    pred_days = 1
    given_days = 1

    save_path = os.path.join('data', save_name)
    data_path = os.path.join(save_path, 'Data.h5')

    train_dataset = h5py.File(data_path, 'r')
    data = np.array(train_dataset['data'])
    print(data.dtype,data.shape)

    input_size = window_size-stride_size


    # For gridsearch 5:1:1
    # Split data
    train_start = 0
    train_end = data.shape[0] - 365*stride_size*3
    valid_start = data.shape[0] - 365*stride_size*3 - input_size
    valid_end = data.shape[0] - 365*stride_size*2
    test_start = data.shape[0] - 365*stride_size*2 - input_size
    test_end = data.shape[0] - 365*stride_size*1
    train_data = data[train_start:train_end,:]
    valid_data = data[valid_start:valid_end,:]
    test_data = data[test_start:test_end,:]
    # Standardlize data
    st_scaler = StandardScaler()
    st_scaler.fit(train_data)
    train_data = st_scaler.transform(train_data)
    valid_data = st_scaler.transform(valid_data)
    test_data = st_scaler.transform(test_data)
    data_scale = st_scaler.scale_
    data_mean = st_scaler.mean_
    assert (np.allclose(train_data * data_scale + data_mean, data[train_start:train_end,:]) == True)
    assert (np.allclose(valid_data * data_scale + data_mean, data[valid_start:valid_end,:]) == True)
    assert (np.allclose(test_data * data_scale + data_mean, data[test_start:test_end,:]) == True)
    # Prepare data
    prep_data_search(train_data, name='train')
    prep_data_search(valid_data, name='valid')
    prep_data_search(test_data, name='test')

    # For inference 6:1:1
    # Split data
    train_start = 0#14600
    train_end = data.shape[0] - 365*stride_size*2
    valid_start = data.shape[0] - 365*stride_size*2 - input_size
    valid_end = data.shape[0] - 365*stride_size*1
    test_start = data.shape[0] - 365*stride_size*1 - input_size
    test_end = data.shape[0]
    train_data = data[train_start:train_end,:]
    valid_data = data[valid_start:valid_end,:]
    test_data = data[test_start:test_end,:]
    # Standardlize data
    st_scaler = StandardScaler()
    st_scaler.fit(train_data)
    train_data = st_scaler.transform(train_data)
    valid_data = st_scaler.transform(valid_data)
    test_data = st_scaler.transform(test_data)
    data_scale = st_scaler.scale_
    data_mean = st_scaler.mean_
    assert (np.allclose(train_data * data_scale + data_mean, data[train_start:train_end,:]) == True)
    assert (np.allclose(valid_data * data_scale + data_mean, data[valid_start:valid_end,:]) == True)
    assert (np.allclose(test_data * data_scale + data_mean, data[test_start:test_end,:]) == True)
    # Prepare data
    # prep_data(train_data[0:20000], name='train')
    prep_data(train_data, name='train')
    prep_data(valid_data, name='valid')
    prep_data(test_data, name='test')


    visualize(data,666,15)



