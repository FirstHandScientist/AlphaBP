import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
import pickle
from joblib import Parallel, delayed
from scipy.stats import multivariate_normal


# importing from src
import sys
sys.path.append("./src")
from modules import MMSE, AlphaBP, MMSEalphaBP, ML, LoopyBP
from utils import channel_component, sampling_noise, sampling_signal, sampling_H,real2complex


# define the progress bar to show the progress


# configuration
class hparam(object):
    num_tx = 4
    num_rx = 4
    soucrce_prior = [0.5, 0.5]
    signal_var = 1
    snr = np.linspace(1, 40, 10)
    monte = 5000
    power_n = 4./3
    constellation = [int(-1), int(1)]

    EC_beta = 0.2
    alpha = None
    ############## observer effect of Alpha for linear system  ########
    algos = {"MMSE": {"detector": MMSE, "legend": "MMSE"},
             "ML": {"detector": ML, "legend": "MAP"},
             "LoopyBP": {"detector": LoopyBP, "legend": "BP"},
             "AlphaBP, 0.3": {"detector": AlphaBP, "alpha": 0.3, "legend":r'$\alpha$-BP, 0.3'},
             "AlphaBP, 0.5": {"detector": AlphaBP, "alpha": 0.5, "legend":r'$\alpha$-BP, 0.5'},
             "AlphaBP, 0.7": {"detector": AlphaBP, "alpha": 0.7, "legend":r'$\alpha$-BP, 0.7'},
             "AlphaBP, 0.9": {"detector": AlphaBP, "alpha": 0.9, "legend":r'$\alpha$-BP, 0.9'}
    }
    ########### import test with pysudo prior ################
    
    # algos = {"MMSE": {"detector": MMSE, "legend": "MMSE"},
    #          "ML": {"detector": ML, "legend": "MAP"},
    #          "LoopyBP": {"detector": LoopyBP, "legend": "BP"},
    #          "MMSEalphaBP, 0.2": {"detector": MMSEalphaBP, "alpha": 0.2, "legend":r'$\alpha$-BP+MMSE, 0.2'},
    #          "MMSEalphaBP, 0.4": {"detector": MMSEalphaBP, "alpha": 0.4, "legend":r'$\alpha$-BP+MMSE, 0.4'},
    #          "MMSEalphaBP, 0.6": {"detector": MMSEalphaBP, "alpha": 0.6, "legend":r'$\alpha$-BP+MMSE, 0.6'},
    #          "MMSEalphaBP, 0.8": {"detector": MMSEalphaBP, "alpha": 0.8, "legend":r'$\alpha$-BP+MMSE, 0.8'},
    # }
    
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
            if key is "MMSE" or key is "ML":
                #### mes detection
                detector = method["detector"](hparam)
                power_ratio = noise_var/hparam.signal_var
                estimated_symbol = detector.detect(y=noised_signal, channel=channel, power_ratio=power_ratio)
                #estimated_symbol = real2complex(np.sign(detected_by_mmse))
            else:
                if "Alpha" in key or "alpha" in key:
                    hparam.alpha = method['alpha']

                detector = method['detector'](noise_var, hparam)
                detector.fit(channel=channel,
                             noise_var=noise_var,
                             noised_signal=noised_signal,
                             stop_iter=50)
                
                        
                estimated_symbol = detector.detect_signal_by_mean()



            est_complex_symbol = real2complex(estimated_symbol)
            error = np.sum(true_symbol != est_complex_symbol)
            
            tmp[key].append(error)

    performance = {"snr": snr}
    for key, method in hparam.algos.items():
        #method["ser"].append( np.mean(tmp[key])/hparam.num_tx )
        performance[key] =  np.mean(np.array(tmp[key]))/hparam.num_tx

 
    return performance

results = []
def collect_result(result):
    global results
    results.append(result)

pool = mp.Pool(mp.cpu_count())


# RESULTS = Parallel(n_jobs=1, pre_dispatch="all", verbose=11, backend="threading")(map(delayed(worker), list(hparam.snr)))
# for snr in list(hparam.snr):
#     pool.apply_async(task, args=(snr), callback=collect_result)
# task(hparam.snr[1])
results = pool.map(task, list(hparam.snr))


#results = [r for r in result_objects]

pool.close()


performance = defaultdict(list)

#for the_result in RESULTS:
for snr in list(hparam.snr):
    for the_result in results:
        if the_result["snr"] == snr:
            for key, _ in hparam.algos.items():                
                performance[key].append( the_result[key] )

    
    
    
# for snr in hparam.snr:

# save the experimental results    
with open("figures/alpha_compare.pkl", 'wb') as handle:
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
fig.savefig("figures/alpha_compare.pdf")
#plt.show()

        
        

