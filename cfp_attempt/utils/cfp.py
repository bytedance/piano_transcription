# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 21:54:18 2017

@author: lisu
"""
import os 
import sys
import soundfile as sf
import numpy as np
import scipy 
from scipy import signal
import argparse
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def STFT(x, fr, fs, Hop, h):        
    t = np.arange(Hop, np.ceil(len(x)/float(Hop))*Hop, Hop)
    N = int(fs/float(fr))
    window_size = len(h)
    f = fs*np.linspace(0, 0.5, np.round(N/2).astype('int'), endpoint=True)
    Lh = int(np.floor(float(window_size-1) / 2))
    tfr = np.zeros((int(N), len(t)), dtype=np.float)     
        
    for icol in range(0, len(t)):
        ti = int(t[icol])           
        tau = np.arange(int(-min([round(N/2.0)-1, Lh, ti-1])), \
                        int(min([round(N/2.0)-1, Lh, len(x)-ti])))
        indices = np.mod(N + tau, N) + 1
        tfr[indices-1, icol] = x[ti+tau-1] * h[Lh+tau-1] \
                                /np.linalg.norm(h[Lh+tau-1])           
                            
    tfr = abs(scipy.fftpack.fft(tfr, n=N, axis=0))  
    return tfr, f, t, N

def nonlinear_func(X, g, cutoff):
    cutoff = int(cutoff)
    if g!=0:
        X[X<0] = 0
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
        X = np.power(X, g)
    else:
        X = np.log(X)
        X[:cutoff, :] = 0
        X[-cutoff:, :] = 0
    return X

def Freq2LogFreqMapping(tfr, f, fr, fc, tc, NumPerOct):
    StartFreq = fc
    StopFreq = 1/tc
    Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break

    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest-1, len(f)), dtype=np.float)
    for i in range(1, Nest-1):
        l = int(round(central_freq[i-1]/fr))
        r = int(round(central_freq[i+1]/fr)+1)
        #rounding1
        if l >= r-1:
            freq_band_transformation[i, l] = 1
        else:
            for j in range(l, r):
                if f[j] > central_freq[i-1] and f[j] < central_freq[i]:
                    freq_band_transformation[i, j] = (f[j] - central_freq[i-1]) / (central_freq[i] - central_freq[i-1])
                elif f[j] > central_freq[i] and f[j] < central_freq[i+1]:
                    freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])
    tfrL = np.dot(freq_band_transformation, tfr)
    return tfrL, central_freq

def Quef2LogFreqMapping(ceps, q, fs, fc, tc, NumPerOct):
    StartFreq = fc
    StopFreq = 1/tc
    Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
    central_freq = []

    for i in range(0, Nest):
        CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break
    f = 1/(q+1e-9)
    Nest = len(central_freq)
    freq_band_transformation = np.zeros((Nest-1, len(f)), dtype=np.float)
    for i in range(1, Nest-1):
        for j in range(int(round(fs/central_freq[i+1])), int(round(fs/central_freq[i-1])+1)):
            if f[j] > central_freq[i-1] and f[j] < central_freq[i]:
                freq_band_transformation[i, j] = (f[j] - central_freq[i-1])/(central_freq[i] - central_freq[i-1])
            elif f[j] > central_freq[i] and f[j] < central_freq[i+1]:
                freq_band_transformation[i, j] = (central_freq[i + 1] - f[j]) / (central_freq[i + 1] - central_freq[i])
    
    tfrL = np.dot(freq_band_transformation[:, :len(ceps)], ceps)
    return tfrL, central_freq

def CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave):
    NumofLayer = np.size(g)

    [tfr, f, t, N] = STFT(x, fr, fs, Hop, h)
    tfr = np.power(abs(tfr), g[0])
    tfr0 = tfr # original STFT
    ceps = np.zeros(tfr.shape)


    if NumofLayer >= 2:
        for gc in range(1, NumofLayer):
            if np.remainder(gc, 2) == 1:
                tc_idx = round(fs*tc)
                ceps = np.real(np.fft.fft(tfr, axis=0))/np.sqrt(N)
                ceps = nonlinear_func(ceps, g[gc], tc_idx)
            else:
                fc_idx = round(fc/fr)
                tfr = np.real(np.fft.fft(ceps, axis=0))/np.sqrt(N)
                tfr = nonlinear_func(tfr, g[gc], fc_idx)
    
    tfr0 = tfr0[:int(round(N/2)),:]
    tfr = tfr[:int(round(N/2)),:]
    ceps = ceps[:int(round(N/2)),:]
    

    HighFreqIdx = int(round((1/tc)/fr)+1)
    f = f[:HighFreqIdx]
    tfr0 = tfr0[:HighFreqIdx,:]
    tfr = tfr[:HighFreqIdx,:]
    HighQuefIdx = int(round(fs/fc)+1)
    q = np.arange(HighQuefIdx)/float(fs)
    ceps = ceps[:HighQuefIdx,:]
    
    tfrL0, central_frequencies = Freq2LogFreqMapping(tfr0, f, fr, fc, tc, NumPerOctave)
    tfrLF, central_frequencies = Freq2LogFreqMapping(tfr, f, fr, fc, tc, NumPerOctave)
    tfrLQ, central_frequencies = Quef2LogFreqMapping(ceps, q, fs, fc, tc, NumPerOctave)

    return tfrL0, tfrLF, tfrLQ, f, q, t, central_frequencies 

def parallel_extract(x, samples, MaxSample, fr, fs, Hop, h, fc, tc, g, NumPerOctave):
    freq_width = MaxSample * Hop
    Round = np.ceil(samples/MaxSample).astype('int')
    tmpL0, tmpLF, tmpLQ, tmpZ = {}, {}, {}, {}
    
    max_workers = min(os.cpu_count(), Round)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_segment = {}
        for i in range(Round):
            tmpX = x[i*freq_width:(i+1)*freq_width]
            future = executor.submit(CFP_filterbank, tmpX, fr, fs, Hop, h, fc, tc, g, NumPerOctave)
            future_to_segment[future] = i

        for future in concurrent.futures.as_completed(future_to_segment):
            seg_id = future_to_segment[future]
            try:
                tfrL0, tfrLF, tfrLQ, f, q, t, CenFreq = future.result()
                tmpL0[seg_id] = tfrL0
                tmpLF[seg_id] = tfrLF
                tmpLQ[seg_id] = tfrLQ
                tmpZ[seg_id] = tfrLF*tfrLQ
            except Exception as exc:
                print("Something generated an exception: {}".format(exc))
                raise exc
    
    return tmpL0, tmpLF, tmpLQ, tmpZ, f, q, t, CenFreq
    
def feature_extraction(
        filename,
        hop=0.02, # in seconds
        w=7939,
        fr=2.0,
        fc=27.5,
        tc=1/4487.0,
        g=[0.24, 0.6, 1],
        NumPerOctave=48,
        Down_fs=44100
    ):
                       
    x, fs = sf.read(filename)
    if len(x.shape)>1:
       x = np.mean(x, axis = 1)
    x = signal.resample_poly(x, Down_fs, fs)
    fs = Down_fs # sampling frequency
    Hop = round(Down_fs*hop)
    x = x.astype('float32')
    h = scipy.signal.blackmanharris(w) # window size
    g = np.array(g)

    MaxSample = 2000
    samples = np.floor(len(x)/Hop).astype('int')
    # print("# Sample: ", samples)
    if samples > MaxSample:
        tmpL0, tmpLF, tmpLQ, tmpZ, f, q, t, CenFreq = parallel_extract(x, samples, MaxSample, fr, fs, Hop, h, fc, tc, g, NumPerOctave)

        tfrL0 = tmpL0.pop(0)
        tfrLF = tmpLF.pop(0)
        tfrLQ = tmpLQ.pop(0)
        Z = tmpZ.pop(0)
        rr = len(tmpL0)
        for i in range(1, rr+1, 1):
            tfrL0 = np.concatenate((tfrL0, tmpL0.pop(i)), axis=1)
            tfrLF = np.concatenate((tfrLF, tmpLF.pop(i)), axis=1)
            tfrLQ = np.concatenate((tfrLQ, tmpLQ.pop(i)), axis=1)
            Z = np.concatenate((Z, tmpZ.pop(i)), axis=1)
    else:
        tfrL0, tfrLF, tfrLQ, f, q, t, CenFreq = CFP_filterbank(x, fr, fs, Hop, h, fc, tc, g, NumPerOctave)
        Z = tfrLF * tfrLQ

    return Z, tfrL0, tfrLF, tfrLQ, t, CenFreq, f

def patch_extraction(Z, patch_size, th):
    # Z is the input spectrogram or any kind of time-frequency representation
    M, N = np.shape(Z)    
    half_ps = int(np.floor(float(patch_size)/2))

    Z = np.append(np.zeros([M, half_ps]), Z, axis = 1)
    Z = np.append(Z, np.zeros([M, half_ps]), axis = 1)
    Z = np.append(Z, np.zeros([half_ps, N+2*half_ps]), axis = 0)

    M, N = np.shape(Z)
    
#    data = np.zeros([1, patch_size, patch_size])
#    mapping = np.zeros([1, 2])
    data = np.zeros([300000, patch_size, patch_size])
    mapping = np.zeros([300000, 2])
    counter = 0
    for t_idx in range(half_ps, N-half_ps):
        PKS, LOCS = findpeaks(Z[:,t_idx], th)
#        print('time at: ', t_idx)
        for mm in range(0, len(LOCS)):
            if LOCS[mm] >= half_ps and LOCS[mm] < M - half_ps and counter<300000:# and PKS[mm]> 0.5*max(Z[:,t_idx]):
                patch = Z[np.ix_(range(LOCS[mm]-half_ps, LOCS[mm]+half_ps+1), range(t_idx-half_ps, t_idx+half_ps+1))]
                patch = patch.reshape(1, patch_size, patch_size)
#                data = np.append(data, patch, axis=0)
#                mapping = np.append(mapping, np.array([[LOCS[mm], t_idx]]), axis=0)
                data[counter,:,:] = patch
                mapping[counter,:] = np.array([[LOCS[mm], t_idx]])
                counter = counter + 1
            elif LOCS[mm] >= half_ps and LOCS[mm] < M - half_ps and counter>=300000:
                print('Out of the biggest size. Please shorten the input audio.')
                
    data = data[:counter-1,:,:]
    mapping = mapping[:counter-1,:]
    Z = Z[:M-half_ps,:]
#    print(data.shape)
#    print(mapping.shape)
    return data, mapping, half_ps, N, Z

def contour_prediction(mapping, pred, N, half_ps, Z, t, CenFreq, max_method):
    PredContour = np.zeros(N)

    pred = pred[:,1]
    pred_idx = np.where(pred>0.5)
    MM = mapping[pred_idx[0],:]
#    print(MM.shape)
    pred_prob = pred[pred_idx[0]]
#    print(pred_prob.shape)
    MM = np.append(MM, np.reshape(pred_prob, [len(pred_prob),1]), axis=1)
    MM = MM[MM[:,1].argsort()]    
    
    for t_idx in range(half_ps, N-half_ps):
        Candidate = MM[np.where(MM[:,1]==t_idx)[0],:]
#        print(Candidate[:,2])
        if Candidate.shape[0] >= 2:
            if max_method == 'posterior':
                fi = np.where(Candidate[:,2]==np.max(Candidate[:,2]))
                fi = fi[0]
            elif max_method == 'prior':
                fi = Z[Candidate[:,0].astype('int'),t_idx].argmax(axis=0)
            fi = fi.astype('int')
#            print(fi)
            PredContour[Candidate[fi,1].astype('int')] = Candidate[fi,0] 
        elif Candidate.shape[0] == 1:
            PredContour[Candidate[0,1].astype('int')] = Candidate[0,0] 
    
    # clip the padding of time
    PredContour = PredContour[range(half_ps, N-half_ps)]
    
    for k in range(len(PredContour)):
        if PredContour[k]>1:
            PredContour[k] = CenFreq[PredContour[k].astype('int')]
    
    Z = Z[:, range(half_ps, N-half_ps)]
#    print(t.shape)
#    print(PredContour.shape)    
    result = np.zeros([t.shape[0],2])
    result[:,0] = t/16000.0
    result[:,1] = PredContour
    return result

def contour_pred_from_raw(Z, t, CenFreq):
    PredContour = Z.argmax(axis=0)
    for k in range(len(PredContour)):
        if PredContour[k]>1:
            PredContour[k] = CenFreq[PredContour[k].astype('int')]
    result = np.zeros([t.shape[0],2])
    result[:,0] = t/16000.0
    result[:,1] = PredContour
    return result

def show_prediction(mapping, pred, N, half_ps, Z, t):
    postgram = np.zeros(Z.shape)
    pred = pred[:,1]
    for i in range(pred.shape[0]):
        postgram[mapping[i,0].astype('int'), mapping[i,1].astype('int')] = pred[i]
    return postgram

def findpeaks(x, th):
    # x is an input column vector
    M = x.shape[0]
    pre = x[1:M - 1] - x[0:M - 2]
    pre[pre < 0] = 0
    pre[pre > 0] = 1

    post = x[1:M - 1] - x[2:]
    post[post < 0] = 0
    post[post > 0] = 1

    mask = pre * post
    ext_mask = np.append([0], mask, axis=0)
    ext_mask = np.append(ext_mask, [0], axis=0)
    
    pdata = x * ext_mask
    pdata = pdata-np.tile(th*np.amax(pdata, axis=0),(M,1))
    pks = np.where(pdata>0)
    pks = pks[0]
    
    locs = np.where(ext_mask==1)
    locs = locs[0]
    return pks, locs

def create_batches(feature, b_size, timesteps, feature_num=384):
    frms = np.ceil(len(feature) / timesteps)
    bss = np.ceil(frms / b_size).astype('int')
    
    pb = (feature_num-feature.shape[1]) // 2
    pt = feature_num-feature.shape[1]-pb
    l = len(feature)
    ch = feature.shape[2]
    pbb = np.zeros((l, pb, ch))
    ptt = np.zeros((l, pt, ch))
    feature = np.hstack([pbb, feature, ptt])

    BSS = []
    for i in range(bss):
        bs = np.zeros((b_size, timesteps, feature.shape[1], feature.shape[2]))
        for ii in range(b_size):
            start_i = i*b_size*timesteps + ii*timesteps
            if start_i >= len(feature):
                break
            end_i = min(start_i+timesteps, len(feature))
            length = end_i - start_i
            
            part = feature[start_i:start_i+length]
            bs[ii, 0:length] = part
        BSS.append(bs)
    
    return BSS
