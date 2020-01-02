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
    
    pd.date_range(ts_data_raw.index[0], periods = 5, freq = 'D', 
                            tz = 'US / Central')
    
    # Select appropriate columns and rename them
    ts_data = ts_data_raw[value_cols]
    ts_data.rename( dict( zip( value_cols, [prefix+x for x in value_cols]) ), axis = 1, inplace=True )
    
    return ts_data
    
    
def series_to_json(ts, target_col):
    '''Returns a dictionary of values in DeepAR, JSON format.
    
       ts: A single time series.
       target_col: col name for target series (other series will be used as dynamic feature)
       
       return: A dictionary of values with "start", "target" and dynamic feature as the rest of the columns
       '''
    # your code here
    remain_col = []
    for col in ts.columns.values:
        if col!= target_col:
            remain_col.append(col)
    
    json_obj = {'start' : ts.index[0].strftime('%Y-%m-%d %H:%M:%S'), 
                'target' : ts[target_col].values.tolist(),
                'dynamic_feat' : ts[remain_col].values.transpose().tolist()}
    return json_obj

class DeepARPredictor(sagemaker.predictor.RealTimePredictor):

    def set_prediction_parameters(self, freq, prediction_length, target_col):
        """Set the time frequency and prediction length parameters. This method **must** be called
        before being able to use `predict`.
        
        Parameters:
        freq -- string indicating the time frequency
        prediction_length -- integer, number of predicted time points
        
        Return value: none.
        """
        self.freq = freq
        self.prediction_length = prediction_length
        self.target_col = target_col
        
    def predict(self, ts, cat=None, encoding="utf-8", num_samples=100, quantiles=["0.1", "0.5", "0.9"]):
        """Requests the prediction of for the time series listed in `ts`, each with the (optional)
        corresponding category listed in `cat`.
        
        Parameters:
        ts -- list of `pandas.Series` objects, the time series to predict
        cat -- list of integers (default: None)
        encoding -- string, encoding to use for the request (default: "utf-8")
        num_samples -- integer, number of samples to compute at prediction time (default: 100)
        quantiles -- list of strings specifying the quantiles to compute (default: ["0.1", "0.5", "0.9"])
        
        Return value: list of `pandas.DataFrame` objects, each containing the predictions
        """
        prediction_times = [x.index[-1]+1 for x in ts]
        req = self.__encode_request(ts, cat, encoding, num_samples, quantiles)
        res = super(DeepARPredictor, self).predict(req)
        return self.__decode_response(res, prediction_times, encoding)
    
    def __encode_request(self, ts, cat, encoding, num_samples, quantiles):
        instances = self.series_to_jsons(ts, self.target_col)
        configuration = {"num_samples": num_samples, "output_types": ["quantiles"], "quantiles": quantiles}
        http_request_data = {"instances": instances, "configuration": configuration}
        return json.dumps(http_request_data).encode(encoding)
    
    def __decode_response(self, response, prediction_times, encoding):
        response_data = json.loads(response.decode(encoding))
        list_of_df = []
        for k in range(len(prediction_times)):
            prediction_index = pd.DatetimeIndex(start=prediction_times[k], freq=self.freq, periods=self.prediction_length)
            list_of_df.append(pd.DataFrame(data=response_data['predictions'][k]['quantiles'], index=prediction_index))
        return list_of_df
    
    def series_to_jsons(ts_s, target_col):
        '''Returns a dictionary of values in DeepAR, JSON format.
    
           ts: A list of single time series data frames
           target_col: col name for target series (other series will be used as dynamic feature)
       
           return: A dictionary of values with "start", "target" and dynamic keys.
        '''
        
        json_objs = []
        for ts in ts_s:
            json_objs.append(series_to_json(ts, target_col))
        return json_objs
