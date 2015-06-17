import binuralfeature as bf
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

filename = "/scratch/Datasets/AVASM/annotated_white_noise/Recorded/1.wav"
#filename = "/scratch/Python/BinuralAudio/sf2_cln.wav"
frq, snd = bf.readaudio(filename)
le = snd.shape[0]

#bf.plotintime(frq,snd,2)
NFFT =  int(frq*0.02) # 20ms window
noverlap = int(frq*0.01) #10ms window
Pxx, freqs, bins, _ = bf.computespectrogram(snd[:,0],frq,nfft=NFFT,noverlap=noverlap, mode='complex')
plt.xlim([0,le/frq])
plt.ylim([0,frq/2])
show()

new_frq = 16000
dnw_snd = bf.downsample(snd[:,0],frq,new_frq)
print dnw_snd.shape
le = dnw_snd.shape[0]

NFFT =  int(new_frq*0.02) # 20ms window
noverlap = int(new_frq*0.01) #10ms window
Pxx, freqs, bins, _ = bf.computespectrogram(dnw_snd,new_frq,nfft=NFFT,noverlap=noverlap, mode='complex')
plt.xlim([0,le/new_frq])
plt.ylim([0,new_frq/2])
show()
