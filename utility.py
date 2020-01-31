import os
import numpy as np
import pandas as pd
import zipfile
import shutil # to delete directory
import datetime
import json
import os # and os for saving



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



def read_ts_data(file_path, skiprows = 3, skipfooter = 2, time_col = "Date and time", value_cols = ["Mean"], prefix = "", freq='H', quality_col = "Quality", accepted_quality = [9,10]):
    '''
    Read the time series csv data and return a pandas data frame with two variables Time and Value
    file_path: csv file path
    
    skiprows: number of top rows to skip
    skipfooter: number of bottom rows to skip
    time_col: name of the timestamp column
    value_cols: name of the target value column
    quality_col: name of the Quality column
    accepted_quality: Accepted quality values.
    '''
    
    # Read csv and remove bottom two rows
    ts_data_raw = pd.read_csv(file_path, skiprows = skiprows, skipfooter = skipfooter, 
                              parse_dates={'DateTime' : [time_col]}, infer_datetime_format=True,
                              index_col='DateTime', engine = 'python')
    
    # Remove 
#     ts_data_raw.index = pd.date_range(ts_data_raw.index[0], periods = ts_data_raw.shape[0], freq = freq)
    
    # Remove rows with un acceptable quality
    ts_data_raw = ts_data_raw[ts_data_raw[quality_col].isin(accepted_quality)]
    
    # Select appropriate columns and rename them
    ts_data = ts_data_raw[value_cols]
    
#     ts_data = ts_data[ts_data[quality_col] in accepted_quality]
    ts_data.rename( dict( zip( value_cols, [prefix+x for x in value_cols]) ), axis = 1, inplace=True )
    
    return ts_data
    
def clean_ts_data(ts, time_delta = datetime.timedelta(hours=1), min_length = 0, min_rain = 10):
    '''
    Breaks the time series into a list of time series when ever there is a break of more than 
    specified difference (time_delta) in the index.
    It will discard time series of length less than min_length
    ts : dataframe with index as datetime
    time_delta : max time delta. 
    min_length : minimum length of the time series to keep
    min_rain : minimum flow data to include
    '''
    clean_ts = ts.dropna()
    series_ids = np.cumsum(np.where(np.diff(clean_ts.index.to_pydatetime()) > time_delta, 1, 0))

    unique_series_ids = np.unique(series_ids)
    
    ts_s = []

    for series_id in unique_series_ids:
        selec_ts = clean_ts.iloc[series_ids == series_id]
        if selec_ts.shape[0] > min_length and  max(selec_ts.iloc[:,1]) >= min_rain:
            ts_s.append(selec_ts)
    
    return ts_s
    
def series_to_json(ts, target_col, prediction_length = 0, cat = None):
    '''Returns a dictionary of values in DeepAR, JSON format.
    
       ts: one time series data frame.
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
    
    if cat is not None:
        json_obj['cat'] = str(cat)
        
    return json_obj

# import json for formatting data

def write_json_dataset(time_series, target_col, filename, cat = None): 
    with open(filename, 'ab') as f:
        # for each of our times series, there is one JSON line
        for ts in time_series:
            json_line = json.dumps(series_to_json(ts, target_col, cat = cat)) + '\n'
            json_line = json_line.encode('utf-8')
            f.write(json_line)
    print(filename + ' saved.')
