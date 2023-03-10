def _window_stack(a, stepsize, width):
    import numpy as np
    return np.hstack([a[i:1+i-width or None:stepsize] for i in range(0,width)])
    
def apply_window_size_on_time_series(inputs, 
                        targets,
                        input_window_size:int = 1,
                        output_window_size:int = 1,
                        offset:int = 0,
                        sequence_stride:int = 1,
                        sampling_rate:int = 1,
                        seed:int = 10,
                        shuffle:bool = False,
                        batch_size:int = None,
                        start_index:int = None,
                        end_index: int = None, 
                        execution_mode:str = "forecast",
                        **kwargs):
    
    """
    This method gets a inputs/targets dataframe and applies window size on them so that
    you can use w_inputs/w_targets to train a model or predict with the model.
    The method accepts differnt input and output window sizes.
    The parameter 'execution_mode' can be set to "training" or "forecast".
    In "training" mode, the method accepts two dataframes: inputs and targets, and returns 
    w_inputs and w_targets. But in "forecast" mode, it ignores the targets 
    and only returns w_inputs.

    :param inputs: this is a dataframe which contains features and it's shape is (n,features).
    :param targets: this is a dataframe which contains targets. It suppurts multi target operations.
                    It's shape is (n,targets).
    :param input_window_size: 
    :param output_window_size: 
    :param offset: this indicates the lag between input and output
                                        windows. If this is 0, both windows start from the same level.
    :param sequence_stride: this identifies the movement step between two successive windows.
    :param sampling_rate: Not supported in this version.
    :param seed:,
    :param shuffle: If shuffle is True, to keep the indices correct, same reordering will be happen 
                    to inputs and targets.
    :param batch_size: batch_size. if None, the method returns all data in one batch.
    :param start_index: this is usefull in "forecast" mode which windowing will be applied on a part of data and
                        not the whole dataframe.
    :param end_index: this is usefull in "forecast" mode which windowing will be applied on a part of data and
                        not the whole dataframe. 
    :param execution_mode: accepts two different modes: "forecast" and "training". If training, the method expects
                        both inputs and targets, and start/end index are not necessary. If forecast, the method expects
                        only inputs, and start/end index should be passed. 
    
                                        
                                        
                                        
    :returns: In training mode, the method returns two numpy ndarray, 
            w_inputs: (batch_size, input_window_size, number_of_features) which is windowed inputs, 
            and 
            w_targets: (batch_size, output_window_size, number_of_targets) which is windowed targets.
            Remember if any of the window_sizes are equal to 1, the dimention will be removed.
            
    :raises ValueError: If input arguments Inputs and Targets have different length in training mode.
    :raises ValueErroe: If sampling_rate is higher than one.
    :raises ValueError: If the rule: (output_window_size + offset >= input_window_size) is not satisfied.
    :raises ValueError: If The rule: (offset >= 0) is not satisfied.
    """

    import pandas as pd
    import numpy as np
        # Truncate if start_index and end_index
        # Sampling is not supported in this version
        # Apply window sizes (considering stride)
        # Shuffle
        # Apply batch_size
                        
    # To keep the operations uniform, convert the input arguments to pandas dataframe if
    # they are pandas series.
    if isinstance(inputs, pd.Series):
        inputs = inputs.to_frame()
    if isinstance(targets, pd.Series):
        targets = targets.to_frame()

    # Trim inputs and targets if 
    if start_index is not None:
        inputs = inputs.iloc[start_index:,:]
        if targets is not None:
            targets = targets.iloc[start_index:,:]
    if end_index is not None:
        inputs = inputs.iloc[:end_index,:]
        if targets is not None:
            targets = targets.iloc[:end_index,:]

    # Initialize the variables
    w_targets = []
    w_inputs = []

    if 'train' in execution_mode.lower():
               
        # The length of the input arguments "inputs" and "targets" expected to be the same,
        if len(inputs) != len(targets):
            raise ValueError('In training mode, Inputs and Targets should have the same length.')
            
        # Sampling is not supported yet,
        if  sampling_rate > 1 :
                raise ValueError('Sampling is not supported in this version...')
        
        # In 'train' mode, we need to truncate inputs and targets to align historical and future edge 
        # of them but since in forecast mode, the method just accepts 'inputs' and not 'targets',
        # there is no need to alignment.

        # Truncate the last rows of the inputs if needed.
        # Check see if future_edge(end of the window) of input/output windows are at the same level or not:
        # If they are the same, no need to truncate last rows of the inputs, O.W. we have to truncate last 
        # rows of the inputs.
        level_of_future_edge_of_in_out_windows_are_different = True if \
                                    (offset +
                                     output_window_size !=\
                                     input_window_size) else False
        if level_of_future_edge_of_in_out_windows_are_different:
            if output_window_size + offset < input_window_size:
                raise ValueError('The rule: (output_window_size + offset >= input_window_size) is not satisfied.')
            else:
                number_of_rows_to_be_cut_from_end_of_the_inputs = offset + output_window_size - input_window_size # last row adresses with -1 so we add one

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
            if offset < 0:
                raise ValueError('The rule: (offset >= 0) is not satisfied.')
            else:
                number_of_rows_to_be_cut_from_beginning_of_the_targets = offset
            targets = targets.iloc[number_of_rows_to_be_cut_from_beginning_of_the_targets:]
            print('number_of_rows_that_are_cut_from_beginning_of_the_targets')
            print(number_of_rows_to_be_cut_from_beginning_of_the_targets)
        
        # Now apply window size on targets. resahpe if single target.
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        w_targets = _window_stack(targets, stepsize=sequence_stride, width=output_window_size)


    # Now apply window size on inputs. resahpe if single input.
    if inputs.ndim == 1:
        inputs = inputs.reshape(-1, 1)
    w_inputs = _window_stack(inputs, stepsize=sequence_stride, width=input_window_size)

    
    # Only apply shuffle in train mode. Check if length of w_inputs and w_targets are equal:
    if ('train' in execution_mode) and shuffle:
        assert len(w_inputs) == len(w_targets)
        idx = np.random.permutation(len(w_inputs))
        w_inputs = w_inputs[idx]
        w_targets = w_targets[idx]

    # Only apply shuffle in train mode. Check if length of w_inputs and w_targets are equal:
    if ('train' in execution_mode) and batch_size:
        number_of_batches = len(w_inputs)/batch_size
        w_inputs = np.vsplit(w_inputs, number_of_batches)
        w_targets = np.vsplit(w_targets, number_of_batches)

    return w_inputs, w_targets
