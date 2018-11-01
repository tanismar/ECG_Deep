import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
import wfdb
import pywt
import pickle as pk
from collections import Counter
from PIL import Image
from PIL import ImageOps

if len(sys.argv)!=(2+1):
    print("Usage: python preprocess_2D.py <NLRAV/NSVFQ> <random_seed>")
    exit(-1)

mode = sys.argv[1] # NLRAV / NSVFQ
r_seed = int(sys.argv[2])

data_names_DS1 = ['101','106', '108', '109', '112', '114', '115', '116', 
                  '118', '119', '122', '124', '201', '203', '205', '207',
                  '208', '209', '215', '220', '223', '230']
data_names_DS2 = ['100', '103', '105', '111', '113', '117', '121', '123',
                  '200', '202', '210', '212', '213', '214', '219', '221', 
                  '222', '228', '231', '232', '233', '234']

wid = 100

if mode=='NLRAV':
    labels = ['N', 'L', 'R', 'A', 'V']
    X = []
    Y = []
    for d in data_names:
        r=wfdb.rdrecord('./data/'+d)
        ann=wfdb.rdann('./data/'+d, 'atr', return_label_elements=['label_store', 'symbol'])
        sig = np.array(r.p_signal[:,0])
        sig_len = len(sig)
        sym = ann.symbol        
        pos = ann.sample
        beat_len = len(sym)
        for i in range(1,beat_len-1):
            if sym[i] in labels: 
                if (pos[i]-pos[i-1])>200 and (pos[i+1]-pos[i])>200:
                    a = sig[pos[i]-150:pos[i]+150]
                    a, cD3, cD2, cD1 = pywt.wavedec(a, 'db6', level=3)
                    X.append(a)
                    Y.append(labels.index(sym[i]))

elif mode=='NSVFQ':
    labels = ['N', 'S', 'V', 'F', 'Q']
    sub_labels = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
    sub = {'N':'N', 'L':'N', 'R':'N', 'e':'N', 'j':'N', 
           'A':'S', 'a':'S', 'J':'S', 'S':'S',
           'V':'V', 'E':'V',
           'F':'F',
           '/':'Q', 'f':'Q', 'Q':'Q'}
    X = []
    Y = []
    for d in data_names:
        r=wfdb.rdrecord('./data/'+d)
        ann=wfdb.rdann('./data/'+d, 'atr', return_label_elements=['label_store', 'symbol'])        
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

data_len = len(X)
np.random.seed(r_seed)
idx = list(range(data_len))
np.random.shuffle(idx)

data_len = int(data_len/5)
idx = idx[:data_len]

_X_train = X
Y_train = Y

print(_X_train.shape)
print(Counter(Y_train))

# Change 1D signal to 2D image
x = list(range(X.shape[1]))
cnt = 0

X_train = None
for i in range(len(_X_train)):
    a = _X_train[i]
    plt.clf()
    plt.figure(figsize=(0.4,0.4))
    plt.plot(x,a)    
    plt.axis('off')
    fn =  labels[Y[i]]+str(i)+'.png'    
    plt.savefig(labels[Y[i]]+'/'+fn)    
    plt.close()
    #img = Image.open(fn).convert("L")
    #img = ImageOps.invert(img)
    #arr = np.asarray(img)
    #arr = np.expand_dims(arr, axis=0)
    #arr = np.expand_dims(arr, axis=-1)
    #X_train = arr.copy() if X_train is None else np.concatenate((X_train,arr), axis=0)
    cnt += 1
    if cnt%1000==0:
        print(cnt)


print(X_train.shape)

#fn = "dataall_2D_"+mode+".pk"
#with open(fn, "wb") as fw:
#    pk.dump(X_train, fw, protocol=pk.HIGHEST_PROTOCOL)
#    pk.dump(Y_train, fw, protocol=pk.HIGHEST_PROTOCOL)
