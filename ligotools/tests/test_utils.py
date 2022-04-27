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

#Set up general variables
time = time_L1
fs = 4096
fband = [43.0, 300.0]
dt = time[1] - time[0]
NFFT = 4*fs
Pxx_L1, freqs = mlab.psd(strain_L1, Fs = fs, NFFT = NFFT)
psd_L1 = interp1d(freqs, Pxx_L1)
deltat_sound = 2.
tevent = 1126259462.44
eventname = 'GW150914' 
bb, ab = butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
normalization = np.sqrt((fband[1]-fband[0])/(fs/2))
strain_L1_whiten = lg.whiten(strain_L1,psd_L1,dt)
strain_L1_whitenbp = filtfilt(bb, ab, strain_L1_whiten) / normalization
indxd = np.where((time >= tevent-deltat_sound) & (time < tevent+deltat_sound))


def test_whiten():
	#test function
	assert type(strain_L1_whiten) is np.ndarray
	assert len(strain_L1_whiten) == 131072
	assert np.mean(strain_L1_whiten) == -0.013534214208369824
	assert np.std(strain_L1_whiten) == 3.9590876402985793



def test_write_wavfile():
	from scipy.io import wavfile
	lg.write_wavfile(eventname+"_L1_whitenbp.wav",int(fs), strain_L1_whitenbp[indxd])
	file = wavfile.read("./audio/" + eventname + "_L1_whitenbp.wav")
	assert file[0] == 4096
	assert type(file[1]) is np.ndarray
	os.remove("./audio/" + eventname + "_L1_whitenbp.wav")

def test_reqshift():
	output = lg.reqshift(strain_L1_whitenbp,fshift=400,sample_rate=fs)
	assert type(output) is np.ndarray
	assert max(output) == 893.136864010276
	assert np.mean(output) == 1.734723475976807e-18
	assert np.std(output) == 8.436672642258083


def test_plot():
	#Load additional variables for plotting function
	fnjson = "./data/BBH_events_v3.json"
	events = json.load(open(fnjson,"r"))
	event = events['GW150914']
	fn_template = './data/' + event['fn_template']  # File name for template waveform
	f_template = h5py.File(fn_template, "r")
	template_p, template_c = f_template["template"][...]
	plottype = "png"
	pcolor = 'g'

	NFFT = 4*fs
	psd_window = np.blackman(NFFT)
	# and a 50% overlap:
	NOVL = NFFT/2

	# define the complex template, common to both detectors:
	template = (template_p + template_c*1.j) 
	# We will record the time where the data match the END of the template.
	etime = time+16
	# the length and sampling rate of the template MUST match that of the data.
	datafreq = np.fft.fftfreq(template.size)*fs
	df = np.abs(datafreq[1] - datafreq[0])

	# to remove effects at the beginning and end of the data stretch, window the data
	# https://en.wikipedia.org/wiki/Window_function#Tukey_window
	try:   dwindow = signal.tukey(template.size, alpha=1./8)  # Tukey window preferred, but requires recent scipy version 
	except: dwindow = signal.blackman(template.size)          # Blackman window OK if Tukey is not available

	# prepare the template fft.
	template_fft = np.fft.fft(template*dwindow) / fs

	# loop over the detectors
	dets = ['L1']
	for det in dets:

		if det is 'L1': data = strain_L1.copy()

		# -- Calculate the PSD of the data.  Also use an overlap, and window:
		data_psd, freqs = mlab.psd(data, Fs = fs, NFFT = NFFT, window=psd_window, noverlap=NOVL)

		# Take the Fourier Transform (FFT) of the data and the template (with dwindow)
		data_fft = np.fft.fft(data*dwindow) / fs

		# -- Interpolate to get the PSD values at the needed frequencies
		power_vec = np.interp(np.abs(datafreq), freqs, data_psd)

		# -- Calculate the matched filter output in the time domain:
		# Multiply the Fourier Space template and data, and divide by the noise power in each frequency bin.
		# Taking the Inverse Fourier Transform (IFFT) of the filter output puts it back in the time domain,
		# so the result will be plotted as a function of time off-set between the template and the data:
		optimal = data_fft * template_fft.conjugate() / power_vec
		optimal_time = 2*np.fft.ifft(optimal)*fs

		# -- Normalize the matched filter output:
		# Normalize the matched filter output so that we expect a value of 1 at times of just noise.
		# Then, the peak of the matched filter output will tell us the signal-to-noise ratio (SNR) of the signal.
		sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
		sigma = np.sqrt(np.abs(sigmasq))
		SNR_complex = optimal_time/sigma

		# shift the SNR vector by the template length so that the peak is at the END of the template
		peaksample = int(data.size / 2)  # location of peak in the template
		SNR_complex = np.roll(SNR_complex,peaksample)
		SNR = abs(SNR_complex)

		# find the time and SNR value at maximum:
		indmax = np.argmax(SNR)
		timemax = time[indmax]
		SNRmax = SNR[indmax]

		# Calculate the "effective distance" (see FINDCHIRP paper for definition)
		# d_eff = (8. / SNRmax)*D_thresh
		d_eff = sigma / SNRmax
		# -- Calculate optimal horizon distnace
		horizon = sigma/8

		# Extract time offset and phase at peak
		phase = np.angle(SNR_complex[indmax])
		offset = (indmax-peaksample)

		# apply time offset, phase, and d_eff to template 
		template_phaseshifted = np.real(template*np.exp(1j*phase))    # phase shift the template
		template_rolled = np.roll(template_phaseshifted,offset) / d_eff  # Apply time offset and scale amplitude

		# Whiten and band-pass the template for plotting
		template_whitened = lg.whiten(template_rolled,interp1d(freqs, data_psd),dt)  # whiten the template
		template_match = filtfilt(bb, ab, template_whitened) / normalization # Band-pass the template

	#Test plot function
	lg.plot(time, timemax, SNR, pcolor, det, eventname, tevent, strain_L1_whitenbp, template_match,
         template_fft, datafreq, d_eff, freqs, data_psd, fs, plottype)
	assert os.path.exists('./figures/' + eventname+"_"+det+"_matchfreq."+plottype)
	os.remove('./figures/' + eventname+"_"+det+"_matchfreq."+plottype)
	assert os.path.exists('./figures/' + eventname+"_"+det+"_matchtime."+plottype)
	os.remove('./figures/' + eventname+"_"+det+"_matchtime."+plottype)
	assert os.path.exists('./figures/' + eventname+"_"+det+"_SNR."+plottype)
	os.remove('./figures/' + eventname+"_"+det+"_SNR."+plottype)

test_whiten()
test_write_wavfile()
test_reqshift()
test_plot()
print("All utils tests ran successfully!")