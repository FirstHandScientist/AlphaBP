# run alpha-bp with graph at different level of sparsity of loops/edges

import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict

from joblib import Parallel, delayed
from scipy.stats import multivariate_normal
import sys


# import the methods and sources
sys.path.append("./src")
from loopy_modules import LoopyBP, AlphaBP, ML
from utils import channel_component, sampling_noise, sampling_signal, sampling_H,real2complex, ERsampling_S, converge_cond_m, messages_to_norm_ratio



# configuration
class hparam(object):
    num_tx = 16
    num_rx = 16
    soucrce_prior = [0.5, 0.5]
    
    stn_var= None
    connect_prob = np.linspace(0.0, 0.9, 10)
    monte = 100
    
    constellation = [int(-1), int(1)]

    alpha = None
    # algos = {"AlphaBP, 0.5": {"detector": AlphaBP, "alpha": 0.5},
    #          "AlphaBP, 1": {"detector": AlphaBP, "alpha": 1},
    #          "AlphaBP, 1.2": {"detector": AlphaBP, "alpha": 1.2}
    # }
    algos = {"AlphaBP, 0.5": {"detector": AlphaBP, "alpha": 0.5, "legend": r'$\alpha=$,'+' {}'.format(0.5)}
    }
    
    # total number of iterations
    iter_num = 200
    # the number of iterations before each each checkpoint
    check_skip = 5
    
    for _, value in algos.items():
        value["ratio"] = []



    

if __name__ == "__main__":
    usage = "python bin/converge_rate.py stn guarantee_cnvg"
    if len(sys.argv) !=3 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(usage)
        sys.exit(1)

    stn = float(sys.argv[1])
    erp = 0.2
    guarantee_cnvg = True if sys.argv[2] == 'true' else False

    for monte in range(hparam.monte):
        for key, method in hparam.algos.items():
            # sampling the S and b for exponential function
            hparam.stn_var = stn
            S, b = ERsampling_S(hparam, erp)
            # regenerate graph if guarantee_cnvg is true that requires convergence
            if guarantee_cnvg:
                while not converge_cond_m(S, method["alpha"]):
                    S, b = ERsampling_S(hparam, erp)

            print("[Convergence: {} at {}]".format(converge_cond_m(S, method["alpha"]), monte))
    
            hparam.alpha = method['alpha']
            detector = method['detector'](None, hparam)
            # set the potential 
            detector.set_potential(S=S, b=b)
            # initialize the messages
            nodes = detector.graph._sorted_nodes()
            detector.graph.init_messages(nodes)

            # do belief propagation: collect belief states every check_skip iterations
            messages_vs_iter = [ detector.lbp_iteration(check_skip=hparam.check_skip,
                                                        stop_iter=hparam.iter_num)
                                 for _ in range(int(hparam.iter_num / hparam.check_skip))]
            method["ratio"].append(messages_to_norm_ratio(messages_vs_iter))
            

    # plot the results
    marker_list = ["o", "<", "+", ">", "v", "1", "2", "3", "8", "*", "h", "d", "D"]
    iter_marker_list = iter(marker_list)
    fig, ax = plt.subplots()
    for key, method in hparam.algos.items():
        # plot the average convergence first
        ax.plot(range(0, hparam.iter_num - hparam.check_skip, hparam.check_skip),
                    np.array(method["ratio"]).mean(axis=0),
                    marker=next(iter_marker_list),
                    label=method['legend'])
        ax.fill_between(range(0, hparam.iter_num - hparam.check_skip, hparam.check_skip),
                        np.array(method["ratio"]).min(axis=0),
                        np.array(method["ratio"]).max(axis=0),
                        alpha=0.5)
    
    ax.set(xlabel="Iteration", ylabel=r'$\frac{\Vert\mathbf{m}^{(n)} - \mathbf{m}^{\ast}\Vert_2}{\Vert \mathbf{m}^{\ast} \Vert_2}$')
    # ax.legend()
    ax.grid()
    ax.set_yscale("log")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.savefig("figures/converge_erp{}_alpha_{}_stn_{}_vs_filter_{}.pdf".format(
        str(erp).replace(".","_"),
        str(hparam.algos.itervalues().next()["alpha"]).replace(".","_"),
        str(stn).replace(".","_"),
        sys.argv[2]))
    
    plt.show()
    # 1. done one step of lbp of the graph
    # 2. call graph.print_messages to get messages in graphs
    # 3. parsing the messages into vector z 

    # 4. record one step of z 
    # 5. used converged z to get the rate convergence curve


