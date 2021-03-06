import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
import wfdb
import pywt

from collections import Counter
from PIL import Image
from PIL import ImageOps



data_names_DS1 = ['101','106', '108', '109', '112', '114', '115', '116', 
                  '118', '119', '122', '124', '201', '203', '205', '207',
                  '208', '209', '215', '220', '223', '230']
data_names_DS2 = ['100', '103', '105', '111', '113', '117', '121', '123',
                  '200', '202', '210', '212', '213', '214', '219', '221', 
                  '222', '228', '231', '232', '233', '234']

labels = ['N', 'S', 'V', 'F', 'Q']
sub_labels = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
sub = {'N':'N', 'L':'N', 'R':'N', 'e':'N', 'j':'N', 
        'A':'S', 'a':'S', 'J':'S', 'S':'S',
        'V':'V', 'E':'V',
        'F':'F',
        '/':'Q', 'f':'Q', 'Q':'Q'}
X = []
Y = []
for d in data_names_DS2:
    r=wfdb.rdrecord('./mitbih/'+d)
    ann=wfdb.rdann('./mitbih/'+d, 'atr', return_label_elements=['label_store', 'symbol'])        
    sig = np.array(r.p_signal[:,0])
    sig_len = len(sig)
    sym = ann.symbol        
    pos = ann.sample
    beat_len = len(sym)
    for i in range(1,beat_len-1):
        if sym[i] in sub_labels: 
            if (pos[i]-pos[i-1])>200 and (pos[i+1]-pos[i])>200:
                a = sig[pos[i]-150:pos[i]+150]
                a, cD3, cD2, cD1 = pywt.wavedec(a, 'db6', level=3)
                X.append(a)
                Y.append(labels.index(sub[sym[i]]))

X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)
print(Counter(Y))

_X_train = X
Y_train = Y

print(_X_train.shape)
print(Counter(Y_train))

# Change 1D signal to 2D image
# x = list(range(X.shape[1]))

#X_train = None
cnt = 0
#for i in range(len(_X_train)):
#    a = _X_train[i]
#    plt.clf()
#    plt.figure(figsize=(2.24,2.24), dpi= 100)   # So the output has size 224 x 224 pixels
#    plt.plot(x,a)    
#    plt.axis('off')
#    fn =  labels[Y[i]]+str(i)+'.png'    
#    plt.savefig('./DS2/'+labels[Y[i]]+'/'+fn)    
#    plt.close()
#    cnt += 1
#    if cnt%1000==0:
#        print(cnt)
print(_X_train.shape)

# Compute wavelet and plot in 2D
cnt = 0
fs = 360
sampling_period = 1 / fs
# for i in range(len(_X_train)):
# Define signal

t = np.linspace(0, 2, 2 * fs)
x = _X_train[i]

# Calculate continuous wavelet transform
coef, freqs = pywt.cwt(x, np.arange(1, 50), 'morl', sampling_period=sampling_period)

# Show w.r.t. time and frequency
plt.figure(figsize=(5, 2))
#plt.pcolormesh(t, freqs, coef)

# Set yscale, ylim and labels
#plt.yscale('log')
#plt.ylim([1, 100])
#plt.ylabel('Frequency (Hz)')
#plt.xlabel('Time (sec)')
# plt.savefig('egg.png', dpi=150)

plt.matshow(coef)
plt.show()


