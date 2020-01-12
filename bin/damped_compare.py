import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
import itertools
from collections import defaultdict
import pickle
from joblib import Parallel, delayed
from scipy.stats import multivariate_normal
from functools import partial
import time
# importing from src
import sys
sys.path.append("./src")
from modules import MMSE, AlphaBP, MMSEalphaBP, ML, LoopyBP, DampBP, TreeReweightBP, NaiveMF
from utils import channel_component, sampling_noise, sampling_signal, sampling_H,real2complex
from utils import step_rate_decay


# define the progress bar to show the progress


# configuration
class hparam(object):
    timing = True
    num_tx = 4
    num_rx = 4
    soucrce_prior = [0.5, 0.5]
    signal_var = 1
    snr = np.linspace(1, 40, 10)
    monte = 100
    power_n = 4./3
    constellation = [int(-1), int(1)]

    EC_beta = 0.2
    alpha = None
    TBP_rate  = 0.25 # the computed appearance probability of an edges in tree reweighted graphs
    ############## observer effect of Alpha for linear system  ########
    algos = {"MMSE": {"detector": MMSE, "legend": "MMSE"},
             "ML": {"detector": ML, "legend": "MAP"},
             "NMF": {"detector": NaiveMF, "legend": "MF"},
             "LoopyBP": {"detector": LoopyBP, "legend": "BP"},
             "TreeReweightBP": {"detector": TreeReweightBP, "legend": "TBP"},
             "DampBP, 0.5": {"detector": DampBP, \
                                      "eta": 0.5, \
                                      "legend":r'Damped-BP'},
             "AlphaBP, 0.5": {"detector": AlphaBP, "alpha": 0.5, "legend":r'$\alpha$-BP'},
             "MMSEalphaBP, 0.5": {"detector": MMSEalphaBP, "alpha": 0.5, "legend":r'$\alpha$-BP+MMSE'},

    }
    
    iter_num = {"EP": 10,
                "EC": 50,
                "LoopyBP": 50,
                "PPBP": 50,
                "AlphaBP": 50,
                "MMSEalphaBP": 50,
                "VariationalBP":50,
                "EPalphaBP": 50,
                "MMSEvarBP":50,
                "LoopyMP": 50}
    
    for _, value in algos.items():
        value["ser"] = []


#pbar = tqdm(total=len(list(hparam.snr)))

def task(snr):

    tmp = dict()
    for name,_ in hparam.algos.items():
        tmp[name] = []

    #progress = tqdm(range(hparam.monte))
    for monte in tqdm(range(hparam.monte)):
        x, true_symbol = sampling_signal(hparam)
        #noise variance in control by SNR in DB
        noise, noise_var = sampling_noise(hparam=hparam, snr=snr)
        channel = sampling_H(hparam)
        noised_signal = np.dot(channel,x) + noise
        for key, method in hparam.algos.items():
            start_time = time.time()
            if key is "MMSE" or key is "ML":
                #### mes detection
                detector = method["detector"](hparam)
                power_ratio = noise_var/hparam.signal_var
                estimated_symbol = detector.detect(y=noised_signal, channel=channel, power_ratio=power_ratio)
                #estimated_symbol = real2complex(np.sign(detected_by_mmse))
            else:
                if key.startswith("AlphaBP,") or key.startswith("MMSEalphaBP,"):
                    hparam.alpha = method['alpha']
                elif key.startswith("AnnealAlphaBP,"):
                    hparam.alpha = method['alpha'][0]
                    hparam.alpha_schedule = partial(step_rate_decay,
                                                    init_lr=method['alpha'][0],
                                                    end=method['alpha'][1],
                                                    anneal_rate=method['anneal_rate'],
                                                    anneal_interval=1)
                elif key.startswith("DampBP,"):
                    hparam.eta = method['eta']


                detector = method['detector'](noise_var, hparam)
                detector.fit(channel=channel,
                             noise_var=noise_var,
                             noised_signal=noised_signal,
                             stop_iter=100)
                
                        
                estimated_symbol = detector.detect_signal_by_mean()



            est_complex_symbol = real2complex(estimated_symbol)
            error = np.sum(true_symbol != est_complex_symbol)
            
            end_time = time.time()
            if not hparam.timing:
                tmp[key].append(error)
            else:
                tmp[key].append(end_time - start_time)


    performance = {"snr": snr}
    for key, method in hparam.algos.items():
        #method["ser"].append( np.mean(tmp[key])/hparam.num_tx )
        if not hparam.timing:
            performance[key] =  np.mean(np.array(tmp[key]))/hparam.num_tx
        else:
            performance[key] =  tmp[key]


 
    return performance

if __name__ == "__main__":
    
    # resutls = [task(hparam.snr[2])]
    # results = [{"snr": snr, "model":task(snr) } for snr in list(hparam.snr)]

    pool = mp.Pool(mp.cpu_count())
    results = pool.map(task, list(hparam.snr))
    pool.close()


    performance = defaultdict(list)

    #for the_result in RESULTS:
    for snr in list(hparam.snr):
        for the_result in results:
            if the_result["snr"] == snr:
                for key, _ in hparam.algos.items():                
                    performance[key].append( the_result[key] )



    if hparam.timing:
        perf_timing = {}
        for method, value in performance.items():
            perf_timing[method] = {'mean': np.array(value).mean(), 'std': np.array(value).std()}

        print(pandas.DataFrame.from_dict(perf_timing))
        sys.exit(0)
    # for snr in hparam.snr:

    # save the experimental results    
    with open("figures/mf_tbp_compare.pkl", 'wb') as handle:
        pickle.dump(performance, handle)

    marker_list = ["o", "<", "+", ">", "v", "1", "2", "3", "8", "*", "h", "d", "D"]
    iter_marker_list = iter(marker_list)
    fig, ax = plt.subplots()
    for key, method in hparam.algos.items():
        ax.semilogy(hparam.snr, performance[key],
                    # label = key + "_Iteration:{}".format(hparam.iter_num[key]) if "MMSE" not in key else "MMSE",
                    label = method['legend'],
                    marker=next(iter_marker_list))

    ax.legend(loc="best", fontsize='small', ncol=2)
    ax.set(xlabel="Ratio of Signal to Noise Variance", ylabel="Symbol Error")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    ax.grid()
    fig.savefig("figures/mf_tbp_compare.pdf")
    #plt.show()


        

