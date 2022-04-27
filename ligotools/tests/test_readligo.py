import numpy as np
import ligotools as lg
import os
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
from scipy.io import wavfile
import json
from scipy import signal


#hide warnings
import warnings
warnings.filterwarnings("ignore")

#load data file for testing
filename = os.path.abspath(u'./data/L-L1_LOSC_4_V2-1126259446-32.hdf5')

#load data using lg.loaddata function
strain_L1, time_L1, chan_dict_L1 = lg.loaddata(filename, 'L1')



#test output types of lg.loaddata
def test_loaddata_types():
	assert type(strain_L1) is np.ndarray
	assert type(time_L1) is np.ndarray
	assert type(chan_dict_L1) is dict
	
	
#test output value for dictionary from lg.loaddata
def test_loaddata_values():
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

	assert (np.array([np.allclose(test_dict[k],chan_dict_L1[k]) for k in test_dict]).all())
	

#test output type for lg.dq_channel_to_seglist
def test_dq_channel_to_seglist_type():
	channel = chan_dict_L1[u'BURST_CAT1']
	seglist = lg.dq_channel_to_seglist(channel)
	assert type(seglist) is list
	
#test output type for lg.dq2segs
def test_dq2segs_type():
	channel = chan_dict_L1[u'BURST_CAT1']
	gps_start = time_L1[0]
	seglist = lg.dq2segs(channel, gps_start)
	assert(isinstance(seglist, lg.SegmentList))


test_loaddata_types()
test_loaddata_values()
test_dq_channel_to_seglist_type()
test_dq2segs_type()


print("All readligo tests ran successfully!")

