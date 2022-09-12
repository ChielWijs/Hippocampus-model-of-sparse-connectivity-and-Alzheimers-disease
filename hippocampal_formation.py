import nengo
import nengo.spa as spa
import numpy as np
import os, inspect
import importlib

import LstsqSparseWeights
importlib.reload(LstsqSparseWeights)

from nengo.neurons import LIF

class Hippocampal_Formation(nengo.Network):
    def __init__(self,
                    dimensions=None,
                    seed=None,
                    n_neurons=1000,
                    ECDGdrop=0,
                    ECDGdropmethod="random",
                    ECDGtrans=1,
                    MFdrop=0,
                    MFdropmethod="lowest", # which weights to drop: lowest, highest, random
                    MFtrans=1,
                    VERBOSE=False,
                    label = "HF"):
        super(Hippocampal_Formation, self).__init__(label=label)

        # if (n_neurons is None or dimensions is None) and load_from is None:
        #     error('Either provide load_from or n_neurons and dimensions.')        

        # settings
        self.seed = seed
        self.dimensions = dimensions
        # EC (_in and _out)
        self.EC_n_neurons = n_neurons
        # DG
        self.DG_n_neurons = self.EC_n_neurons * 10   
        # CA3
        self.CA3_n_neurons = self.EC_n_neurons * 2 

        if VERBOSE:
            print(  f"--- CREATING MODEL ---\n" 
                    f"- SETTINGS -\n"
                    f"dimenstionality: {dimensions}, seed: {seed}\n"
                    f"- ENSEMBLE SIZES -\n"
                    f"EC: {self.EC_n_neurons}, DG: {self.DG_n_neurons}, CA3: {self.CA3_n_neurons}\n"
                    f"- CONNECTIVITY -\n"
                    f"ECDGdrop: {ECDGdrop}, ECDGdropmethod: {ECDGdropmethod}, ECDGtrans: {ECDGtrans}\n"
                    f"MFdrop: {MFdrop}, MFdropmethod: {MFdropmethod}, MFtrans: {MFtrans}")    

        with self:

            # a Node to connect an input signal to the entorhinal cortex (EC)
            self.input = nengo.Node(None, size_in=dimensions, label="input node")

            # the EC recieves input from the input node
            self.EC = nengo.Ensemble(self.EC_n_neurons, dimensions, seed=seed, label="EC_in")
            nengo.Connection(self.input, self.EC, synapse=None, seed=seed, label="input -> EC")

            # the EC provides input to the dentate gyrus (DG)
            self.DG = nengo.Ensemble(self.DG_n_neurons, dimensions, intercepts=nengo.dists.Uniform(0,0), seed=seed, label="DG")
            self.ECtoDG =  nengo.Connection(self.EC, self.DG, 
                                                solver=LstsqSparseWeights.LstsqSparseWeights(drop=ECDGdrop, drop_method=ECDGdropmethod, excitatory=False, seed=seed),
                                                transform=ECDGtrans,
                                                seed=seed, label="EC -> DG")

            # the DG provides input to region CA3                                    
            self.CA3 = nengo.Ensemble(self.CA3_n_neurons, dimensions, intercepts=nengo.dists.Uniform(0,0), seed=seed, label="CA3")
            self.DGtoCA3 = nengo.Connection(self.DG, self.CA3, 
                                                solver=LstsqSparseWeights.LstsqSparseWeights(drop=MFdrop, drop_method=MFdropmethod, excitatory=False, seed=seed), 
                                                transform=MFtrans,
                                                seed=seed, label="DG -> CA3")