import numpy as np
from ligotools import readligo as rl
import os

#hide warnings
import warnings
warnings.filterwarnings("ignore")

#load data file for testing
filename = os.path.abspath(u'data/L-L1_LOSC_4_V2-1126259446-32.hdf5')

#test output types of rl.loaddata
def test_loaddata_types():
	strain_L1, time_L1, chan_dict_L1 = rl.loaddata(filename, 'L1')
	assert type(strain_L1) is np.ndarray
	assert type(time_L1) is np.ndarray
	assert type(chan_dict_L1) is dict
	
	
#test output value for dictionary from rl.loaddata
def test_loaddata_values():
	strain_L1, time_L1, chan_dict_L1 = rl.loaddata(filename, 'L1')
	ones_arr = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
	test_dict = {u'BURST_CAT1': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
 u'BURST_CAT2': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
 u'BURST_CAT3': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
 u'CBC_CAT1': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
 u'CBC_CAT2': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
 u'CBC_CAT3': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
 u'DATA': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
 'DEFAULT': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
 u'NO_BURST_HW_INJ': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
 u'NO_CBC_HW_INJ': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
 u'NO_CW_HW_INJ': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
 u'NO_DETCHAR_HW_INJ': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
 u'NO_STOCH_HW_INJ': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)}
	#print(chan_dict_L1)
	#print(test_dict)
	assert (np.array([np.allclose(test_dict[k],chan_dict_L1[k]) for k in test_dict]).all())
	

#test output type for rl.dq_channel_to_seglist
def test_dq_channel_to_seglist_type():
	strain_L1, time_L1, chan_dict_L1 = rl.loaddata(filename, 'L1')
	channel = chan_dict_L1[u'BURST_CAT1']
	seglist = rl.dq_channel_to_seglist(channel)
	assert type(seglist) is list
	
#test output type for rl.dq2segs
def test_dq2segs_type():
	strain_L1, time_L1, chan_dict_L1 = rl.loaddata(filename, 'L1')
	channel = chan_dict_L1[u'BURST_CAT1']
	gps_start = time_L1[0]
	seglist = rl.dq2segs(channel, gps_start)
	assert(isinstance(seglist, rl.SegmentList))


test_loaddata_types()
test_loaddata_values()
test_dq_channel_to_seglist_type()
test_dq2segs_type()


print("All tests ran successfully!")