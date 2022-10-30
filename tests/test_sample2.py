import pandas as pd
import numpy as np
import numpy.testing as npt
import pytest
from timeseries_windowing.apply_window_size_on_timeseries import apply_window_size_on_time_series

d = {'lagged_feature':[np.NAN,10,11,12,13,14,15,16,17], 
     'independent_feature': [10,20,30,40,50,60,70,80,90],
     'target1': [10,11,12,13,14,15,16,17,18],
     'target2': [110,111,112,113,114,115,116,117,118]}
idx = [1,2,3,4,5,6,7,8,9]
df = pd.DataFrame(data=d, index=idx)

# scenario #1: single_feature, single_target, offset=0, training_mode, window_sizes=1
scenario_1_inputs = df[['lagged_feature']]
scenario_1_targets = df.loc[:,'target1']
scenario_1_exp_w_in = [[np.nan], [10.], [11.], [12.], [13.], [14.], [15.], [16.], [17.]]
scenario_1_exp_w_out = [[10], [11], [12], [13], [14], [15], [16], [17], [18]]

# scenario #2: multi_feature, multui_target, offset=1, training_mode, input_window=3, output_window=2
scenario_2_inputs = df.loc[:,['lagged_feature','independent_feature']]
scenario_2_targets = df.loc[:,['target1','target2']]
scenario_2_exp_w_in = [[[np.nan, 10.],  [10., 20.],  [11., 30.]],
                        [[10., 20.],  [11., 30.],  [12., 40.]],
                        [[11., 30.],  [12., 40.],  [13., 50.]],
                        [[12., 40.],  [13., 50.],  [14., 60.]],
                        [[13., 50.],  [14., 60.],  [15., 70.]],
                        [[14., 60.],  [15., 70.],  [16., 80.]],
                        [[15., 70.],  [16., 80.],  [17., 90.]]]

scenario_2_exp_w_out = [[[11, 111],  [12, 112]],
                        [[12, 112],  [13, 113]],
                        [[13, 113],  [14, 114]],
                        [[14, 114],  [15, 115]],
                        [[15, 115],  [16, 116]],
                        [[16, 116],  [17, 117]],
                        [[17, 117],  [18, 118]]]

@pytest.mark.parametrize(
    'inputs, targets, input_window_size, output_window_size, offset, sampling_rate, seed, sequence_stride, shuffle, start_index, end_index, execution_mode, kwargs, exp_w_inputs, exp_w_targets', 
    [
        (scenario_1_inputs, scenario_1_targets, 1, 1, 0, 1, 10, 1, False, None, None, 'training', {}, scenario_1_exp_w_in, scenario_1_exp_w_out),
        (scenario_2_inputs, scenario_2_targets, 3, 2, 1, 1, 10, 1, False, None, None, 'training', {}, scenario_2_exp_w_in, scenario_2_exp_w_out),
    ]
)

def test_sample(inputs, targets, input_window_size, output_window_size, offset, sampling_rate, seed, sequence_stride, shuffle, start_index, end_index, execution_mode, kwargs, exp_w_inputs, exp_w_targets):
    npt.assert_equal(apply_window_size_on_time_series(inputs, 
                                            targets, 
                                            input_window_size, 
                                            output_window_size, 
                                            offset, 
                                            sampling_rate, 
                                            seed, 
                                            sequence_stride, 
                                            shuffle, 
                                            start_index, 
                                            end_index, 
                                            execution_mode, 
                                            **kwargs), [exp_w_inputs,exp_w_targets])
