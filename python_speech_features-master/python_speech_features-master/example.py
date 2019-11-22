#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

for i in range(10):
     (rate,sig) = wav.read("genres/blues/blues.0000"+str(i)+".wav")
    mfcc_feat = mfcc(sig,rate)
    d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig,rate)
    print(fbank_feat[1:2,:])
