import os
import numpy as np
import pandas as pd
import zipfile
import shutil # to delete directory


def unzip_ts_data(data_dir):
    '''
    Unzip all the zip files in the given data directory
    Then, it will return a list of path to all the extracted csv files.

    data_dir: path to directory containing ts zip files
    
    '''

    # Zip data files in Caboolture folder
    zip_files = os.listdir(data_dir)

    # List to store path to the csv files
    csv_paths = [] 

    for zip_file in zip_files:
        if '.zip' in zip_file:
            # Directory name to extract contents of zip file
            zip_file_path = data_dir + '/' + zip_file
            zip_dir = data_dir + '/' + zip_file.split('.')[0]

            print(zip_dir)
            # Delete directory if it exists
            if os.path.exists(zip_dir):
                os.system('rm -r ' + zip_dir) 

            # Create dir and unzip
            os.mkdir(zip_dir)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(zip_dir)
            print('unzipped {}!'.format(zip_file))

            # Save path to csv
            for file_name in os.listdir(zip_dir):
                if '.csv' in file_name:
                    csv_paths.append('{}/{}'.format(zip_dir,file_name))
                    
    return csv_paths



def read_ts_data(file_path, skiprows = 3, skipfooter = 2, time_col = "Date and time", value_cols = ["Mean"], prefix = "", freq='H'):
    '''
    Read the time series csv data and return a pandas data frame with two variables Time and Value
    file_path: csv file path
    
    skiprows: number of top rows to skip
    skipfooter: number of bottom rows to skip
    time_col: name of the timestamp column
    value_cols: name of the target value column
    '''
    
    # Read csv and remove bottom two rows
    ts_data_raw = pd.read_csv(file_path, skiprows = skiprows, skipfooter = skipfooter, 
                              parse_dates={'DateTime' : [time_col]}, infer_datetime_format=True,
                              index_col='DateTime', engine = 'python')
    
    ts_data_raw.index = pd.date_range(ts_data_raw.index[0], periods = ts_data_raw.shape[0], freq = freq)
    
    # Select appropriate columns and rename them
    ts_data = ts_data_raw[value_cols]
    ts_data.rename( dict( zip( value_cols, [prefix+x for x in value_cols]) ), axis = 1, inplace=True )
    
    return ts_data
    
    
def series_to_json(ts, target_col, prediction_length = 0):
    '''Returns a dictionary of values in DeepAR, JSON format.
    
       ts: A single time series.
       target_col: col name for target series (other series will be used as dynamic feature)
       prediction_length: if prediction_length is given, then this number of target variable is removed from the end
       
       return: A dictionary of values with "start", "target" and dynamic feature as the rest of the columns
       '''
    # your code here
    remain_col = []
    for col in ts.columns.values:
        if col!= target_col:
            remain_col.append(col)
    
    n = ts.shape[0]
    
    json_obj = {'start' : ts.index[0].strftime('%Y-%m-%d %H:%M:%S'), 
                'target' : ts[target_col].values[0:(n-prediction_length)].tolist(),
                'dynamic_feat' : ts[remain_col].values.transpose().tolist()}
    return json_obj


