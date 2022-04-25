import numpy as np
from ligotools import readligo as rl
import os

#hide warnings
import warnings
warnings.filterwarnings("ignore")

#load data file for testing
filename = os.path.abspath(u'../data/L-L1_LOSC_4_V2-1126259446-32.hdf5')

#load data using rl.loaddata function
strain_L1, time_L1, chan_dict_L1 = rl.loaddata(filename, 'L1')



#test output types of rl.loaddata
def test_loaddata_types():
	#strain_L1, time_L1, chan_dict_L1 = rl.loaddata(filename, 'L1')
	assert type(strain_L1) is np.ndarray
	assert type(time_L1) is np.ndarray
	assert type(chan_dict_L1) is dict
	
	
#test output value for dictionary from rl.loaddata
def test_loaddata_values():
	#strain_L1, time_L1, chan_dict_L1 = rl.loaddata(filename, 'L1')
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
	#strain_L1, time_L1, chan_dict_L1 = rl.loaddata(filename, 'L1')
	channel = chan_dict_L1[u'BURST_CAT1']
	seglist = rl.dq_channel_to_seglist(channel)
	assert type(seglist) is list
	
#test output type for rl.dq2segs
def test_dq2segs_type():
	#strain_L1, time_L1, chan_dict_L1 = rl.loaddata(filename, 'L1')
	channel = chan_dict_L1[u'BURST_CAT1']
	gps_start = time_L1[0]
	seglist = rl.dq2segs(channel, gps_start)
	assert(isinstance(seglist, rl.SegmentList))


test_loaddata_types()
test_loaddata_values()
test_dq_channel_to_seglist_type()
test_dq2segs_type()


print("All readligo tests ran successfully!")


#Set up variables
time = time_L1
dt = time[1] - time[0]
NFFT = 4*fs
Pxx_L1, freqs = mlab.psd(strain_L1, Fs = fs, NFFT = NFFT)
psd_L1 = interp1d(freqs, Pxx_L1)
deltat_sound = 2.
fs = 4096
tevent = 1126259462.44
eventname = 'GW150914' 
bb, ab = butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
normalization = np.sqrt((fband[1]-fband[0])/(fs/2))
strain_L1_whiten = ut.whiten(strain_L1,psd_L1,dt)
strain_L1_whitenbp = filtfilt(bb, ab, strain_L1_whiten) / normalization
indxd = np.where((time >= tevent-deltat_sound) & (time < tevent+deltat_sound))


def test_whiten():
	#test function
	assert type(strain_L1_whiten) is np.ndarray
	assert len(strain_L1_whiten) = 131072
	assert np.mean(strain_L1_whiten) = -0.013534214208369824
	assert np.std(strain_L1_whiten) = 3.9590876402985793



def test_write_wavfile():
	ut.write_wavfile(eventname+"_L1_whitenbp.wav"+ "test",int(fs), strain_L1_whitenbp[indxd])
	file = wavfile.read(eventname+"_L1_whitenbp.wav" + "test")
	assert file[0] == 4096
	assert type(file[1]) is np.ndarray
	os.remove(eventname+"_L1_whitenbp.wav" + "test")

def test_reqshift():
	output = reqshift(strain_L1_whitenbp,fshift=400,sample_rate=fs)
	assert type(output) is np.ndarray
	



test_whiten()
test_write_wavfile()

print("All utils tests ran successfully!")