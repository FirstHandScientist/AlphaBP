# run alpha-bp with graph at different level of sparsity of loops/edges
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict

from joblib import Parallel, delayed
# from scipy.stats import multivariate_normal
import pickle
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
    connect_prob = np.linspace(0.0, 0.9, 5)
    
    monte = 50
    constellation = [int(-1), int(1)]

    alpha = None
    stn_var= .1
    # algos = {"LoopyBP": {"detector": LoopyBP, "alpha": None},
    # }
    
    algos = {"BP": {"detector": LoopyBP, "alpha": None, "legend": "BP"},
             # "AlphaBP, 0.2": {"detector": AlphaBP, "alpha": 0.2, "legend": r'$\alpha$-BP, 0.2'},
             "AlphaBP, 0.4": {"detector": AlphaBP, "alpha": 0.4, "legend": r'$\alpha$-BP, 0.4'},
             # "AlphaBP, 0.6": {"detector": AlphaBP, "alpha": 0.6, "legend": r'$\alpha$-BP, 0.6'},
             "AlphaBP, 0.8": {"detector": AlphaBP, "alpha": 0.8, "legend": r'$\alpha$-BP, 0.8'},
             "AlphaBP, 1.2": {"detector": AlphaBP, "alpha": 1.2, "legend": r'$\alpha$-BP, 1.2'}

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

    
    
# save the experimental results    
with open("figures/MRF_MAPprob.pkl", 'wb') as handle:
    pickle.dump(performance, handle)

    
# for snr in hparam.snr:


marker_list = ["o", "<", "+", ">", "v", "1", "2", "3", "8"]
iter_marker_list = iter(marker_list)
fig, ax = plt.subplots()
for key, method in hparam.algos.items():
    ax.plot(hparam.connect_prob, 1 - np.array(performance[key]),
            label = method['legend'],
            marker=next(iter_marker_list))
    


ax.legend(loc="best", fontsize='medium', ncol=2)
ax.set_ylim([0.45, 1])
ax.set(xlabel=r"Edge probability $\gamma$ ", ylabel="MAP accuracy")
ax.grid()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.savefig("figures/MAPacc_edgeP_sum.pdf")
# plt.show()

        
        

