def apply_window_size_on_time_series(inputs, 
                        targets,
                        input_window_size:int = 1,
                        output_window_size:int = 1,
                        offset:int = 0,
                        sampling_rate:int = 1,
                        seed:int = 10,
                        sequence_stride:int = 1,
                        shuffle:bool = False,
                        start_index:int = None,
                        end_index: int = None, 
                        execution_mode:str = "forecast",
                        **kwargs):
    
    """
    This method gets a dataframe and applies window size on them so that
    you can feed them into the deep models for trainig and forecasting purposes.
    The method accepts differnt input and output windows for inputs(features) and targets.
    The parameter execution_mode can be set to "training" or "forecast".
    In "training" mode, it accepts two dataframes: inputs and targets, and returns 
    w_inputs and w_targets. But in "forecast" mode, it only accepts inputs dataframe 
    and returns w_inputs.

    :param inputs: this is a dataframe which contains features and it's shape is (n,features).
    :param targets: this is a dataframe which contains targets. It suppurts multi target operations.
                    It's shape is (n,targets).
    :param input_window_size: 
    :param output_window_size: 
    :param offset: this indicates the lag between input and output
                                        windows. If this is 0, both windows start from the same level.
    :param sampling_rate: this will apply on both inputs and targets.
    :param seed:,
    :param sequence_stride: this identifies the movement step between two successive windows.
    :param shuffle: If shuffle is True, to keep the indices correct, same reordering will be happen 
                    to inputs and targets.
    :param start_index: this is usefull in "forecast" mode which windowing will be applied on a part of data and
                        not the whole dataframe.
    :param end_index: this is usefull in "forecast" mode which windowing will be applied on a part of data and
                        not the whole dataframe. 
    :param execution_mode: accepts two different modes: "forecast" and "training". If training, the method expects
                        both inputs and targets, and start/end index are not necessary. If forecast, the method expects
                        only inputs, and start/end index should be passed. 
    
                                        
                                        
                                        
    :returns: In training mode, the method returns two numpy ndarray, 
            w_inputs: (total_number_of_rows, input_window_size, number_of_features) which is windowed inputs, 
            and 
            w_targets: (total_number_of_rows, output_window_size, number_of_targets) which is windowed targets.
            Remember if any of the window_sizes are equal to 1, the dimention will be removed.
            
    :raises ValueError: If input arguments Inputs and Targets have different length in training mode.
    :raises ValueErroe: Sampling is only applicable if both input/output window sizes are higher than 1, 
                    so if any of the window_sizes is 1 and the other is GT 1, and sampling is higher than 1, 
                    an error will be thrown.
    """

    import pandas as pd
    import numpy as np
    import tensorflow as tf
    
    # To keep the operations uniform, convert the input arguments to pandas dataframe if
    # they are pandas series.
    if isinstance(inputs, pd.Series):
        inputs = inputs.to_frame()
    if isinstance(targets, pd.Series):
        targets = targets.to_frame()

    # Initialize the variables
    w_targets = []
    w_inputs = []
    
    # This method doesn't respetc batch_size. Users can apply bach_size as a parameter of fit method. 
    batch_size = inputs.shape[0]

    if 'train' in execution_mode.lower():
        
        # The length of the input arguments "inputs" and "targets" expected to be the same,
        if len(inputs) != len(targets):
            raise ValueError('In training mode, Inputs and Targets should have the same length.')
            
        # Sampling is only applicable if both input and output window sizes are higher than 1, so if any of the window_sizes is 1 and
        # the other is not, and sampling is higher than 1, an error will be thrown,
        if  ((input_window_size == 1 and output_window_size > 1)
            or
            (input_window_size > 1 and output_window_size == 1)) \
            and \
            sampling_rate > 1 :
                raise ValueError('Sampling is only applicable if both input/output window sizes are higher than 1.')
        
        # Check see if future_edge(end of the window) of input/output windows are at the same level or not:
        # If they are the same, no need to truncate end of the inputs, O.W. we have to truncate last rows of the 
        # inputs.
        level_of_future_edge_of_in_out_windows_are_different = True if \
                                    (offset +
                                     output_window_size !=\
                                     input_window_size) else False
        if level_of_future_edge_of_in_out_windows_are_different:
            how_much_onward_is_the_future_edge_of_out_window_vs_in_window = \
                                    (offset + 
                                    output_window_size -
                                    input_window_size)
            if how_much_onward_is_the_future_edge_of_out_window_vs_in_window < 0:
                raise ValueError('offset + output_window_size >= input_window_size')
            number_of_rows_to_be_cut_from_end_of_the_inputs = \
                                    how_much_onward_is_the_future_edge_of_out_window_vs_in_window # last row adresses with -1 so we add one
            inputs = inputs.iloc[:-1*(number_of_rows_to_be_cut_from_end_of_the_inputs)]
            print('number_of_rows_that_are_cut_from_end_of_the_inputs:')
            print(number_of_rows_to_be_cut_from_end_of_the_inputs)

        # Truncate the first rows of the targets if needed.
        # Check see if historical edge of input and output windows are at the same level or not:
        # If they are the same, no need to truncate targets, O.W. we have to truncate first rows of the 
        # targets.
        level_of_historical_edge_of_in_out_windows_are_different = True if \
                                    offset != 0 else False
        if level_of_historical_edge_of_in_out_windows_are_different:
            how_much_onward_is_the_historical_edge_of_out_window_vs_in_window = \
                                    offset
            if how_much_onward_is_the_historical_edge_of_out_window_vs_in_window < 0:
                raise ValueError('Output_window can not start before input_window. offset should be GE 0.')
            number_of_rows_to_be_cut_from_beginning_of_the_targets = \
                                    how_much_onward_is_the_historical_edge_of_out_window_vs_in_window
            targets = targets.iloc[number_of_rows_to_be_cut_from_beginning_of_the_targets:]
            print('number_of_rows_that_are_cut_from_beginning_of_the_targets')
            print(number_of_rows_to_be_cut_from_beginning_of_the_targets)

        if (output_window_size > 1):
            w_targets = tf.keras.preprocessing.timeseries_dataset_from_array(data=targets,
                                        targets=None, 
                                        sequence_length=output_window_size,
                                        sequence_stride=sequence_stride,
                                        sampling_rate=sampling_rate,
                                        batch_size=batch_size,
                                        # shuffle=shuffle,
                                        shuffle=False,
                                        seed=seed,
                                        start_index=start_index,
                                        end_index=end_index)
        else: 
            w_targets = targets.copy()    

    if input_window_size > 1:
        w_inputs = tf.keras.preprocessing.timeseries_dataset_from_array(data=inputs,
                                    targets=None, 
                                    sequence_length=input_window_size,
                                    sequence_stride=sequence_stride,
                                    sampling_rate=sampling_rate,
                                    batch_size=batch_size,
                                    # shuffle=shuffle,
                                    shuffle=False,
                                    seed=seed,
                                    start_index=start_index,
                                    end_index=end_index)
    else:
        w_inputs = inputs.copy()
    if 'forecast' in execution_mode.lower():
        if (start_index is None) or (end_index is None):
            raise ValueError('In Forecast mode, start_index and end_index should be passed.')
        if input_window_size == 1:
            return w_inputs.iloc[start_index:end_index,:].to_numpy(), w_targets
        else:
            return list(w_inputs.as_numpy_iterator())[0], w_targets
    
    if input_window_size == 1:
        w_inputs = w_inputs.to_numpy()
    else:
        w_inputs = list(w_inputs.as_numpy_iterator())[0]

    if output_window_size == 1:
        w_targets = w_targets.to_numpy()
    else:
        w_targets = list(w_targets.as_numpy_iterator())[0]
    return w_inputs, w_targets
