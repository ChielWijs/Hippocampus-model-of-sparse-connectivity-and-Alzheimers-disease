import matplotlib.pyplot as plt
import os, inspect
import nengo.spa as spa
import numpy as np
import nengo.utils.numpy as npext

cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

from sklearn.metrics import jaccard_score

from timeit import default_timer as timer

def calculate_activity_at_SFT(vectors, window=10, minimum=1):
    """
    Parameters
    ----------
        vectors : ndarray
            (number of measurement, number of patterns/stimuli, number of neurons)
        window : int
            collapse activation of neurons over "window" number of time step
            to find similarity across a short timespan iso a single time point.
    """
    # non overlapping addition of every "window" elements, on axis=0 (default) to sum over time
    vecs = np.add.reduceat(vectors, np.arange(0, len(vectors), window))
    # set all non zero elements to 1
    vecs[vecs<(minimum*1000)] = 0 # multiply minimum with 1/dt as a single spike gets a value of 1/dt
    # print(f"for minimum {minimum}, len(vecs[vecs!=0]) -> {len(vecs[vecs!=0])}")
    vecs[vecs!=0] = 1

    aboveSFT = np.array([])
    for step in vecs:
        aboveSFT = np.append(aboveSFT, np.mean(np.add.reduce(step, axis=1))) # count above SFT per pattern, average over patterns, average over timesteps outside loop

    return np.mean(aboveSFT)


def calculate_windowed_jaccard_score(vectors, window=10, minimum=1):
    """
    Parameters
    ----------
        vectors : ndarray
            (number of measurement, number of patterns/stimuli, number of neurons)
        window : int
            collapse activation of neurons over "window" number of time step
            to find similarity across a short timespan iso a single time point.
    """
    # start = timer()
    # non overlapping addition of every "window" elements, on axis=0 (default) to sum over time
    vecs = np.add.reduceat(vectors, np.arange(0, len(vectors), window))
    # set all non zero elements to 1
    vecs[vecs<(minimum*1000)] = 0 # multiply minimum with 1/dt as a single spike gets a value of 1/dt
    # print(f"for minimum {minimum}, len(vecs[vecs!=0]) -> {len(vecs[vecs!=0])}")
    vecs[vecs!=0] = 1

    scores = np.array([])
    # calculate metric between all vectors at a certain "windowed" timestep
    for step in vecs:
        for i in range(len(step)):
            for j in range(i+1,len(step)):
                scores = np.append(scores, jaccard_score(step[i],step[j], zero_division=0.0)) 

    return np.mean(scores)



