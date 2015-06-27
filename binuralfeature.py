from scipy.io import wavfile
import os.path
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import stride_tricks
from scikits.samplerate import resample
from scipy import math
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
	d = gcd(oldrate,newrate)
	new_snd   = resample(snd,newrate/oldrate,'sinc_best')
	return np.array(new_snd)


def myspectrogram(x,nfft=2048,frameSize=512,fs=1,window=np.hamming,noverlapratio=0.5):
	noverlap = int(frameSize*noverlapratio)
	win = window(frameSize)
	M =  len(win); # frameSize
	if len(x)< M: # zero-pad to fill a window
		x = np.hstack((x,np.zeros((M-len(x),))))

	Modd = np.mod(M,2) # O if M is even, 1 if odd
	Mo2 = (M-Modd)/2;
	
	if noverlap < 0:
		nhop = -noverlap
	else:
		nhop = M-noverlap

	nx = len(x)
	nframes = int((nx-noverlap)/nhop)
	X = np.array(np.zeros((nfft,nframes)), dtype=complex) # allocate output spectrogram
	zp = np.zeros((nfft-M,)) # zero padding each FFT
	xframe = np.zeros((M,))
	xoff  = 0 - Mo2 #input time offset = half a frame

	for m in range(0,nframes):
		if xoff < 0:
			xframe[0:xoff+M] = x[0:xoff+M,] # partial input data frame
		else:
			xframe = x[xoff:xoff+M]

		xw= win * xframe # apply window
		xwzp = np.hstack((xw[Mo2:M],zp,xw[1:Mo2]))
		fft_ = np.fft.fft(xwzp,nfft)
		X[:,m] = fft_
		xoff += nhop

	return X 

def feature_ILDandIPD(XL_,XR_):
	ILD_LR = 20*np.log10(abs(XR_)/abs(XL_))
	IPD_LR = np.angle(XR_/np.conj(XL_))
	return (ILD_LR,IPD_LR)


def plotspectrogram(XL,nfft,fs,title='spectrogram'):	
	autopower = np.abs(XL * np.conj(XL))
	result  = 20*np.log10(autopower) 
	#result = np.clip(result, -40, 200)    # clip values

	x = np.arange(0,XL.shape[1])
	y = np.arange(0,XL.shape[0])
	y *= (fs/2)/(nfft/2)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(result, origin='lower', interpolation='bilinear',cmap='jet', aspect='auto')
	ax.set_ylabel('Frequency (Hz)')
	ax.set_title(title)
	# Tweak spacing to prevent clipping of tick-labels
	plt.subplots_adjust(bottom=0.15)
	plt.show()

def plotIPD_ILD(x,type=''):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(x, origin='lower', interpolation='bilinear',cmap='jet', aspect='auto')
	ax.set_ylabel('Frequency (Hz)')
	ax.set_title(type)
	plt.show()

def plotintime(frq,snd,chnum=0):
	le = len(snd)
	timearray = np.arange(0,le*1.0,1)
	timearray = timearray/frq
	timearray = timearray*1000 #scale ot milliseconds
	plt.plot(timearray,snd,color='b')
	plt.ylabel('Amplitude')
	plt.xlabel('Time (ms)')
	plt.xlim([0,le*1000.0/frq])
	plt.show(block=False)

