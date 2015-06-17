from scipy.io import wavfile
import os.path
import numpy as np
import matplotlib
from matplotlib import pylab, mlab
from matplotlib import pyplot as plt
from scikits.samplerate import resample
from fractions import gcd

def extract():
	return (1)

def printinfo(txt):
	print "Info :", txt


def readaudio(filename):	
	#check if the file exist
	if os.path.exists(filename):
		sampRate, snd = wavfile.read(filename)
		if snd.dtype == 'int16':
			snd  = snd/(2.**15)
		elif snd.dtype =='int32':
			snd  = snd/(2.**31)

		return (sampRate,snd)
	else:
		raise ValueError('File does not exists')

""" downsample audio """
def downsample(snd,oldrate,newrate):
	r = gcd(newrate,oldrate)
	new_snd = resample(snd, r,'linear')
	return new_snd

""" short time fourier transform of audio signal """
def computestft(samples,samplerate,winsize=2048,overlapFac=0.5,window=np.hanning):
	s = stft(samples, winsize,overlapFac,window)    
	return s

def computespectrogram(samples,samplerate,nfft=2048,noverlap=0,window=pylab.window_hanning, mode='default'):
	spec,freqs,ts,im = pylab.specgram(samples, NFFT=nfft, Fs=samplerate, window=window, noverlap=noverlap, cmap =None,xextent=None, pad_to=None, sides='onesided',scale_by_freq=None)
	return (spec,freqs,ts,im)


def plotintime(frq,snd,chnum=1):
	le, ch = snd.shape
	timearray = np.arange(0,le*1.0,1)
	timearray = timearray/frq
	timearray = timearray*1000 #scale ot milliseconds
	plt.plot(timearray,snd[:,chnum],color='b')
	plt.ylabel('Amplitude')
	plt.xlabel('Time (ms)')
	plt.xlim([0,le*1000.0/frq])
	plt.show(block=False)

