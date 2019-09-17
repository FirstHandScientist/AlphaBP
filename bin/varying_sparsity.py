# run alpha-bp with graph at different level of sparsity of loops/edges

import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import matplotlib
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict

from joblib import Parallel, delayed
from scipy.stats import multivariate_normal
import sys

sys.path.append("./src")
from loopy_modules import LoopyBP, AlphaBP, ML
from utils import channel_component, sampling_noise, sampling_signal, sampling_H,real2complex, ERsampling_S



# configuration
class hparam(object):
    num_tx = 16
    num_rx = 16
    soucrce_prior = [0.5, 0.5]
    signal_var = 1
    stn_var= 1
    connect_prob = np.linspace(0.0, 0.9, 10)
    monte = 20
    power_n = 4./3
    constellation = [int(-1), int(1)]

    EC_beta = 0.2
    alpha = None
    algos = {"LoopyBP": {"detector": LoopyBP, "alpha": None},
             "AlphaBP, 0.2": {"detector": AlphaBP, "alpha": 0.2},
             "AlphaBP, 0.4": {"detector": AlphaBP, "alpha": 0.4},
             "AlphaBP, 0.6": {"detector": AlphaBP, "alpha": 0.6},
             "AlphaBP, 0.8": {"detector": AlphaBP, "alpha": 0.8},
             "AlphaBP, 1.2": {"detector": AlphaBP, "alpha": 1.2}
             
    }
    iter_num = 50

    
    for _, value in algos.items():
        value["ser"] = []

def task(erp):
    tmp = dict()
    for name,_ in hparam.algos.items():
        tmp[name] = []
    for monte in range(hparam.monte):

        # sampling the S and b for exponential function
        S, b = ERsampling_S(hparam, erp)

        # compute the joint ML detection
        detectML = ML(hparam)
        solution = detectML.detect(S, b)
        
        for key, method in hparam.algos.items():
            hparam.alpha = method['alpha']
            detector = method['detector'](None, hparam)
            detector.fit(S=S,
                         b=b,
                         stop_iter=hparam.iter_num)

            estimated_symbol = detector.detect_signal_by_mean()

            error = np.sum(np.array(solution) != np.array(estimated_symbol))
            tmp[key].append(error)

    # performance should be made by comparing with ML
    performance = {"erp": erp}
    for key, method in hparam.algos.items():
        #method["ser"].append( np.mean(tmp[key])/hparam.num_tx )
        performance[key] =  np.mean(tmp[key])/hparam.num_tx 
    return performance

results = []
def collect_result(result):
    global results
    results.append(result)

if __name__ == "__main__":
    usage = "python bin/varying_sparsity.py"

    pool = mp.Pool(mp.cpu_count())

    results = pool.map(task, list(hparam.connect_prob))
    pool.close()

    performance = defaultdict(list)

    #for the_result in RESULTS:
    for connect_prob in list(hparam.connect_prob):
        for the_result in results:
            if the_result["erp"] == connect_prob:
                for key, _ in hparam.algos.items():                
                    performance[key].append( the_result[key] )

    # for snr in hparam.snr and plot the results
    marker_list = ["o", "<", "+", ">", "v", "1", "2", "3", "8"]
    iter_marker_list = iter(marker_list)
    fig, ax = plt.subplots()
    for key, method in hparam.algos.items():
        ax.plot(hparam.connect_prob, performance[key],
                label = key,
                marker=next(iter_marker_list))


    ax.legend()
    ax.set(xlabel="ERP", ylabel="SER")
    ax.grid()
    fig.savefig("figures/erp_experiment.pdf")
    plt.show()


