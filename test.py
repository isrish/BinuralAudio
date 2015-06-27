import binuralfeature as bf
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

filename = "/scratch/Datasets/AVASM/annotated_white_noise/Recorded/1.wav"
#filename = "/scratch/Python/BinuralAudio/sf2_cln.wav"
frq, snd = bf.readaudio(filename)
print snd.shape

#bf.plotintime(frq,snd,2)
snd1 = snd[:,0];

fs = 16000.
dnw_snd = bf.downsample(snd1,frq,fs)
print 'New downsampled size:', dnw_snd.shape
bf.plotintime(fs,dnw_snd,0)
plt.show()


# Parameter setting
time_win = 64.
frame_shift = 32.
ns=fs/1000*time_win;
nfft=1024;
no=fs/1000*(time_win-frame_shift);
nzp=ns/2;
zp=np.zeros((nzp,));


XL = bf.myspectrogram(np.hstack((zp,dnw_snd,zp)),nfft,ns,fs,np.hamming,noverlapratio=0.5)
nfreq,nframe =XL.shape
nfreq = nfreq/2;
inside = range(int(np.rint(nzp/(ns-no))),int(np.rint(nframe-nzp/(ns-no))))
XL = XL[0:nfreq,inside];
print 'Spectogram Size:', XL.shape


# plot Spectogram
bf.plotspectrogram(XL,nfft,fs,title='CH-Left')



## RIGHT
#bf.plotintime(frq,snd,2)
snd1 = snd[:,1];

dnw_snd = bf.downsample(snd1,frq,fs)
print 'New downsampled size:', dnw_snd.shape
bf.plotintime(fs,dnw_snd,0)
plt.show()


# Parameter setting
time_win = 64.
frame_shift = 32.
ns=fs/1000*time_win;
nfft=1024;
no=fs/1000*(time_win-frame_shift);
nzp=ns/2;
zp=np.zeros((nzp,));


XR= bf.myspectrogram(np.hstack((zp,dnw_snd,zp)),nfft,ns,fs,np.hamming,noverlapratio=0.5)
nfreq,nframe =XR.shape
nfreq = nfreq/2;
inside = range(int(np.rint(nzp/(ns-no))),int(np.rint(nframe-nzp/(ns-no))))
XR = XR[0:nfreq,inside];
print 'Right Spectogram Size:', XR.shape


# plot Spectogram
bf.plotspectrogram(XR,nfft,fs,title='CH-right')




ILD,IPD = bf.feature_ILDandIPD(XL,XR)

bf.plotIPD_ILD(ILD,'ILD')
bf.plotIPD_ILD(IPD,'IPD')











''' max freq recoveed is f2/2 ==8000hz, using nfft=1024 point the frequency 
resolution is fs/2/nfft=7.8hz per frequency bin,
'''