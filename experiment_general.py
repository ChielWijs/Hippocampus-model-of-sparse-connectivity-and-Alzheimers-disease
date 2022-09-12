import numpy as np
import matplotlib.pyplot as plt
from IPython.display import FileLink 
import nengo
import nengo.spa as spa
import importlib #python3
import sys
import numpy
np.set_printoptions(threshold=sys.maxsize)
import inspect
from similarityPlot import *

# from ocl_sim import MyOCLsimulator
import os

import numpy.matlib as matlib
import pandas as pd

import scipy
import scipy.special
import scipy.sparse

import inspect, os, sys, time, csv, random

import png ##pypng
import itertools
import base64
import PIL.Image

import socket
import warnings
import gc
import datetime

import importlib

import spa_hippocampal_formation       
importlib.reload(spa_hippocampal_formation)

import pyopencl
import nengo_ocl
ctx = pyopencl.create_some_context()

os.environ["PYOPENCL_CTX"] = "0"

cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script path

def initialize_vocab(seed=1):
    
    global vocab
    global vocab_all_words
    global vocab_target_pairs

    global target_words

    global target_pairs
    global target_fan1_pairs
    global target_fan2_pairs

    global target_pairs_keys

    target_fan1_pairs = [
            ['HAWK', 'KNOB'], 
            ['PLATE', 'LOOP'], 
            ['MEDAL', 'TOOL'],
            ['MAZE', 'TUBE'],
            ['SCAR', 'VEIL'],
            ['SPEAR', 'BADGE'],
            ['BLOCK', 'CLOUD'],
            ['HORN', 'BRAIN']
    ]

    target_fan2_pairs = [
            ['GUEST', 'JUDGE'],
            ['COIN', 'TANK'],
            ['SODA', 'ATOM'],
            ['ACRE', 'STOVE'],
            ['GUEST', 'ATOM'],
            ['COIN', 'JUDGE'],
            ['SODA', 'STOVE'],
            ['ACRE', 'TANK'] 
    ]

    target_pairs = target_fan1_pairs + target_fan2_pairs

    target_words = [word for pair in target_pairs for word in pair]
    target_words = list(dict.fromkeys(target_words)) # remove duplicates as the spa.Vocabulary does not like that

    rng_vocabs = np.random.RandomState(seed=seed)     
    attempts_vocabs = 2000 

    vocab = spa.Vocabulary(D, max_similarity=.05,rng=rng_vocabs)

    for word in target_words:
        #add semantic pointer for each word
        vocab.add(word, vocab.create_pointer(attempts=attempts_vocabs)) 

    target_pairs_keys = []

    for pair in target_pairs:
        # add semantic pointer for the normalized addition of a target pair
        x1 = vocab.parse(pair[0])
        x2 = vocab.parse(pair[1])
        x = x1*(1/x1.length()) + x2*(1/x2.length())
        vocab.add('%s_%s' % (pair[0], pair[1]), x*(1/x.length())) 
        target_pairs_keys.append('%s_%s' % (pair[0], pair[1]))

    vocab_all_words = vocab.create_subset(target_words)
    vocab_target_pairs = vocab.create_subset(target_pairs_keys)

from timeit import default_timer as timer
from datetime import timedelta

def do_experiment():

    global current_pair
    global df

    print("Start of overlap experiment for subject %s" % subj)

    fans = [1] * 8 + [2] * 8 

    EC_vectors = np.zeros((1000, 16, n_neurons*1)) # number of probe measurements (1s/dt), number of stimuli, number of neurons
    EC_activity = np.zeros((16))    
    
    DG_vectors = np.zeros((1000, 16, n_neurons*10)) # number of probe measurements (1s/dt), number of stimuli, number of neurons
    DG_activity = np.zeros((16))
    
    CA3_vectors = np.zeros((1000, 16, n_neurons*2)) # number of probe measurements (1s/dt), number of stimuli, number of neurons
    CA3_activity = np.zeros((16))

    start = timer()

    for trial, pair in enumerate(target_pairs_keys): # 8 * fan 1 target; 8 * fan 2 target;
        current_pair = pair

        print("subj: %s, trial: %s, fan: %s, stimulus: %s                " % (subj, trial+1, fans[trial], pair), end = "\r")

        sim.run(1)

        EC_data = sim.data[EC_probe]
        DG_data = sim.data[DG_probe].copy()
        CA3_data = sim.data[CA3_probe]

        for i in range(len(CA3_vectors)):
            EC_vectors[i, trial, :] = EC_data[i]
            DG_vectors[i, trial, :] = DG_data[i]
            CA3_vectors[i, trial, :] = CA3_data[i]

        temp = sim.data[EC_probe].copy()
        temp[temp!=0] = 1
        EC_activity[trial] = temp.sum(1).mean() # sum activity per time step, calculate mean spikes for trial

        temp = sim.data[DG_probe].copy()
        temp[temp!=0] = 1
        DG_activity[trial] = temp.sum(1).mean() # sum activity per time step, calculate mean spikes for trial

        temp = sim.data[CA3_probe].copy()
        temp[temp!=0] = 1
        CA3_activity[trial] = temp.sum(1).mean() # sum activity per time step, calculate mean spikes for trial

        sim.reset()

    end = timer()

    print(f"End of experiment for subject {subj}, DURATION: {timedelta(seconds=end-start)}                                ")
    print(f"Start of analysis for subject {subj} data", end="\r")
    startT = timer()

    for start, end, fan in [(0, 8, "fan 1"), (8, 17, "fan 2"), (0, 17, "both")]:
        # Activity
        activityEC = EC_activity[start:end].mean()
        activityDG = DG_activity[start:end].mean()
        activityCA3 = CA3_activity[start:end].mean()

        # Sparsity
        for w in [10, 20, 30]:

            overlapEC = calculate_windowed_jaccard_score(EC_vectors[:,start:end,:], 100, w)
            overlapDG = calculate_windowed_jaccard_score(DG_vectors[:,start:end,:], 100, w)
            overlapCA3 = calculate_windowed_jaccard_score(CA3_vectors[:,start:end,:], 100, w)

            SFTactivityEC = calculate_activity_at_SFT(EC_vectors[:,start:end,:], 100, w) / n_neurons
            SFTactivityDG = calculate_activity_at_SFT(DG_vectors[:,start:end,:], 100, w) / (n_neurons * 10)
            SFTactivityCA3 = calculate_activity_at_SFT(CA3_vectors[:,start:end,:], 100, w) / (n_neurons * 2)    

            df = df.append({"subj":subj, "fan":fan, "ROI":"EC", "ECDGdrop":ECDGdrop, "ECtrans":ECDGtrans, "MFdrop":MFdrop, "MFdropmethod":MFdropmethod, "MFtrans":MFtrans, "minimum":w, "overlap":overlapEC, "activity":activityEC, "SFTactivity":SFTactivityEC}, ignore_index=True) 
            df = df.append({"subj":subj, "fan":fan, "ROI":"DG", "ECDGdrop":ECDGdrop, "ECtrans":ECDGtrans, "MFdrop":MFdrop, "MFdropmethod":MFdropmethod, "MFtrans":MFtrans, "minimum":w, "overlap":overlapDG, "activity":activityDG, "SFTactivity":SFTactivityDG}, ignore_index=True)
            df = df.append({"subj":subj, "fan":fan, "ROI":"CA3", "ECDGdrop":ECDGdrop, "ECtrans":ECDGtrans, "MFdrop":MFdrop, "MFdropmethod":MFdropmethod, "MFtrans":MFtrans, "minimum":w, "overlap":overlapCA3, "activity":activityCA3, "SFTactivity":SFTactivityCA3}, ignore_index=True)


    end=timer()

    print(f"End of analysis for subject {subj} data, DURATION: {timedelta(seconds=end-startT)}                                ")

def prepare_sim(seed=None, progress_bar=False):

    global sim

    print("--- BUILDING SIMULATOR ---", end="\r")

    start = timer()

    sim = nengo_ocl.Simulator(model, context=ctx, seed=fseed, progress_bar=progress_bar)

    end = timer()

    print(f"--- BUILDING SIMULATOR FINISHED. DURATION: {timedelta(seconds=end-start)} ---")


def input_func(t): # input function visual stimuli
    item1, item2 = current_pair.split("_")
    return f"{item1} + {item2}"


def main():
    global model

    global n_neurons
    global D
    global fseed
    global subj

    global current_pair
    current_pair = "0_0"

    global EC_probe
    global DG_probe
    global CA3_probe

    global ECDGdrop
    global ECDGtrans
    global MFdrop
    global MFdropmethod
    global MFtrans

    ECDGdrop = 0
    ECDGtrans = 1
    MFdrop = 0
    MFdropmethod = "lowest"
    MFtrans = 1

    global df

    global sim

    n_neurons = 500
    D = 16
    n_subj = 10

    for i in range(n_subj):
    
        subj = i+1
        fseed = subj
        np.random.seed(fseed)

        initialize_vocab(seed=fseed)
        
        column_names = ["subj", "fan", "ROI", "ECDGdrop", "ECtrans", "MFdrop", "MFdropmethod", "MFtrans", "minimum", "overlap", "activity", "SFTactivity"]
        df = pd.DataFrame(columns = column_names)

        model = spa.SPA(seed=fseed)
        with model:
            ##### Input #####
            model.exp_input = spa.State(D, vocab=vocab, seed=fseed)
            model.input = spa.Input(exp_input=input_func) 

            ##### Hippocampal Formation #####
            model.HF = spa_hippocampal_formation.SPA_Hippocampal_Formation(
                dimensions=D,
                seed=fseed,
                n_neurons=n_neurons,
                ECDGdrop=ECDGdrop,
                ECDGtrans=ECDGtrans,
                MFdrop=MFdrop,
                MFdropmethod=MFdropmethod, # which weights to drop: lowest, highest, random
                MFtrans=MFtrans,
                VERBOSE=True   
            )

            nengo.Connection(model.exp_input.output, model.HF.input)

            EC_probe = nengo.Probe(model.HF.HF.EC.neurons, sample_every=0.001)
            DG_probe = nengo.Probe(model.HF.HF.DG.neurons, sample_every=0.001)
            CA3_probe = nengo.Probe(model.HF.HF.CA3.neurons, sample_every=0.001)                       

            prepare_sim()

            do_experiment()
            
            sim.close()
            del sim
            del model

        try:
            df.to_csv(cur_path + f"/results/general_{subj}.csv")
        except FileNotFoundError as e:
            print('Problem with storing data for separation testing')
            print(e)
        else:
            print('Succesfully stored data for separation testing')


if __name__ == "__main__":
    main()

