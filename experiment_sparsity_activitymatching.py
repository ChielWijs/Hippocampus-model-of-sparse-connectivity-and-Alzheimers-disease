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

def do_experiment(store=True, calculate_activity=False, VERBOSE=True):

    global current_pair
    global df

    global baseline_activity
    global current_activity

    if VERBOSE: print("Start of overlap experiment for subject %s" % subj)

    fans = [1] * 8 + [2] * 8 

    # DG_vectors = np.zeros((1000, 16, n_neurons*10))
    CA3_vectors = np.zeros((1000, 16, n_neurons*2)) # number of probe measurements (1s/dt), number of stimuli, number of neurons
    CA3_activity = np.zeros((16))

    start = timer()

    for trial, pair in enumerate(target_pairs_keys): # 8 * fan 1 target; 8 * fan 2 target;
        current_pair = pair

        if VERBOSE: print("subj: %s, trial: %s, fan: %s, stimulus: %s                " % (subj, trial+1, fans[trial], pair), end = "\r")

        sim.run(1)

        # ECin_data = sim.data[ECin_neurons_probe]
        # DG_data = sim.data[DG_probe].copy()
        CA3_data = sim.data[CA3_probe]

        for i in range(len(CA3_vectors)):
            # DG_vectors[i, trial, :] = DG_data[i]
            CA3_vectors[i, trial, :] = CA3_data[i]
            # ECin_vectors[i, trial, :] = ECin_data[i]

        temp = sim.data[CA3_probe].copy()
        temp[temp!=0] = 1
        CA3_activity[trial] = temp.sum(1).mean() # sum activity per time step, calculate mean spikes for trial

        sim.reset()

    end = timer()

    if VERBOSE: print(f"End of experiment for subject {subj}, DURATION: {timedelta(seconds=end-start)}                                ")
    if VERBOSE: print(f"Start of analysis for subject {subj} data", end="\r")
    startT = timer()

    if calculate_activity:

        if MFdrop == 0: 
            baseline_activity = CA3_activity[1:17].mean()
        else:
            current_activity = CA3_activity[1:17].mean()

    if store:

        for start, end, fan in [(0, 8, "fan 1"), (8, 17, "fan 2"), (0, 17, "both")]:
            # Activity
            activity = CA3_activity[start:end].mean()

            # Sparsity
            for w in [10, 20, 30]:

                # overlapDG = calculate_windowed_jaccard_score(DG_vectors, 100, w)
                overlapCA3 = calculate_windowed_jaccard_score(CA3_vectors[:,start:end,:], 100, w)

                SFTactivityCA3 = calculate_activity_at_SFT(CA3_vectors[:,start:end,:], 100, w) / (n_neurons * 2)

                # df = df.append({"subj":subj, "ROI":"DG", "ECDGdrop":ECDGdrop, "ECtrans":ECtrans, "MFdrop":MFdrop, "MFdropmethod":MFdropmethod, "MFtrans":MFtrans, "minimum":w, "overlap":overlapDG}, ignore_index=True)
                df = df.append({"subj":subj, "fan":fan, "ROI":"CA3", "ECDGdrop":ECDGdrop, "ECtrans":ECDGtrans, "MFdrop":MFdrop, "MFdropmethod":MFdropmethod, "MFtrans":MFtrans, "minimum":w, "overlap":overlapCA3, "activity":activity, "SFTactivity":SFTactivityCA3}, ignore_index=True)

    end=timer()

    if VERBOSE: print(f"End of analysis for subject {subj} data, DURATION: {timedelta(seconds=end-startT)}                                ")

def prepare_sim(seed=None, progress_bar=True):

    global sim

    if progress_bar: print("--- BUILDING SIMULATOR ---                    ", end="\r")

    start = timer()

    sim = nengo_ocl.Simulator(model, context=ctx, seed=fseed, progress_bar=False)

    end = timer()

    if progress_bar: print(f"--- BUILDING SIMULATOR FINISHED. DURATION: {timedelta(seconds=end-start)} ---")


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

    global DG_probe
    global CA3_probe

    global ECDGdrop
    global ECDGtrans
    global MFdrop
    global MFdropmethod
    global MFtrans

    global df

    global sim

    n_neurons = 500
    D = 16
    n_subj = 10

    for i in range(n_subj):

        ECDGdrop = 0
        ECDGtrans = 1
        MFdrop = 0
        MFdropmethod = "lowest"
        MFtrans = 1
    
        subj = i+1
        fseed = subj
        np.random.seed(fseed)

        initialize_vocab(seed=fseed)
        
        column_names = ["subj", "fan", "ROI", "ECDGdrop", "ECtrans", "MFdrop", "MFdropmethod", "MFtrans", "minimum", "overlap", "activity", "SFTactivity"]
        df = pd.DataFrame(columns = column_names)

        # ------- establish activity baseline ------- 
        for mfdropmethod in ["highest", "random", "lowest"]:
            MFdrop = 0
            MFtrans = 1
            MFdropmethod = mfdropmethod

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

                DG_probe = nengo.Probe(model.HF.HF.DG.neurons, sample_every=0.001)
                CA3_probe = nengo.Probe(model.HF.HF.CA3.neurons, sample_every=0.001)                       

            prepare_sim()

            do_experiment(store=True, calculate_activity=True)
            
            sim.close()
            del sim
            del model

            # ------- activity matching ------- 
            for mfdrop in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

                MFdrop = mfdrop
                MFtrans = 1
                trans_step = 32
                fitting = True
                solved = False
                stored = False

                start_fit = timer()

                while ( fitting or solved ) and not stored:
                    
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
                            VERBOSE=False  
                        )

                        nengo.Connection(model.exp_input.output, model.HF.input)

                        DG_probe = nengo.Probe(model.HF.HF.DG.neurons, sample_every=0.001)
                        CA3_probe = nengo.Probe(model.HF.HF.CA3.neurons, sample_every=0.001)                       

                    if fitting:
                        print(f"Try for MFdrop = {MFdrop}, MFmethod = {MFdropmethod}, MFtrans = {MFtrans}", end = "\r")
                        prepare_sim(progress_bar=False)
                        do_experiment(store=False, calculate_activity=True, VERBOSE=False)
                    elif solved:
                        prepare_sim(progress_bar=True)
                        do_experiment(store=True, calculate_activity=True, VERBOSE=True)
                        solved = False
                        stored = True
                    else:
                        error("There is a mistake in the while loop")

                    if abs(current_activity - baseline_activity) <= 0.10 * baseline_activity:
                        # MFtrans stays the same
                        fitting = False
                        solved = True
                    elif current_activity < baseline_activity: # increase until we reach the desired activity level
                        MFtrans += trans_step
                    else: # in case we overshoot
                        trans_step = trans_step / 2
                        MFtrans -= trans_step

                    sim.close()
                    del sim
                    del model

                end_fit = timer()

                print(f"--- TRANSFORM VALUE FITTED: {MFtrans}. BASELINE ACTIVITY: {baseline_activity}, FITTED ACTIVITY: {current_activity}. DURATION: {timedelta(seconds=end_fit-start_fit)} ---")

        try:
            df.to_csv(cur_path + f"/results/dropXmethod_activitymatching_subj_{subj}.csv")
        except FileNotFoundError as e:
            print('Problem with storing data for separation testing')
            print(e)
        else:
            print('Succesfully stored data for separation testing')


if __name__ == "__main__":
    main()

