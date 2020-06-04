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
from utils import channel_component, sampling_noise, sampling_signal, sampling_H,real2complex, ERsampling_S, largest_singular



# configuration
class hparam(object):
    num_tx = 16
    num_rx = 16
    soucrce_prior = [0.5, 0.5]
    
    stn_var= None
    monte = 100
    constellation = [int(-1), int(1)]
    alpha = None
    # algos = {"AlphaBP, 0.5": {"detector": AlphaBP, "alpha": 0.5},
    #          "AlphaBP, 1": {"detector": AlphaBP, "alpha": 1},
    #          "AlphaBP, 1.2": {"detector": AlphaBP, "alpha": 1.2}
    # }
    algos = {"AlphaBP, 0.5, ep0.2": {"detector": AlphaBP, "alpha": 0.5, "erp": 0.2, "legend": r'$\alpha=$'+'{}'.format(0.5)+',$\gamma=$'+'{}'.format(0.2) },
             "AlphaBP, 0.5, ep0.4": {"detector": AlphaBP, "alpha": 0.5, "erp": 0.4, "legend": r'$\alpha=$'+'{}'.format(0.5)+',$\gamma=$'+'{}'.format(0.4) },
             "AlphaBP, 1, ep0.4": {"detector": AlphaBP, "alpha": 1, "erp": 0.4, "legend": r'$\alpha=$'+'{}'.format(1)+',$\gamma=$'+'{}'.format(0.4) },
             "AlphaBP, 1, ep0.2": {"detector": AlphaBP, "alpha": 1, "erp": 0.2, "legend": r'$\alpha=$'+'{}'.format(1)+'$,\gamma=$'+'{}'.format(0.2) }
    }
    
    for _, value in algos.items():
        value["lgst_singular"] = None


    

if __name__ == "__main__":
    usage = "python bin/contract_condition.py"
    if sys.argv[0] == "-h" or sys.argv[0] == "--help":
        print(usage)
        sys.exit(1)
    stn_var_list = np.linspace(0.01, 0.3, 20)
    for key, method in hparam.algos.items():
        svd_largest = []
        for stn_var in stn_var_list:
            tmp = []
            for monte in range(hparam.monte):
                # sampling the S and b for exponential function
                hparam.stn_var = stn_var
                S, b = ERsampling_S(hparam, method['erp'])
                # regenerate graph if guarantee_cnvg is true that requires convergence
                tmp.append(largest_singular(S, method["alpha"]))
            svd_largest.append(np.mean(tmp))

            
        method["lgst_singular"] = np.array(svd_largest)

    # plot the results
    marker_list = ["o", "<", "+", ">", "v", "1", "2", "3", "8", "*", "h", "d", "D"]
    iter_marker_list = iter(marker_list)
    fig, ax = plt.subplots()
    for key, method in hparam.algos.items():
        # plot the average convergence first
        ax.plot(stn_var_list,
                    method["lgst_singular"],
                    marker=next(iter_marker_list),
                    label=method['legend'])
        
    ax.set(xlabel=r"$\sigma$", ylabel=r"$\lambda^{\ast}(\mathbf{M})$")
    ax.legend()
    ax.grid()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.savefig("figures/contraction_vs_variance.pdf")
    plt.show()
    # 1. done one step of lbp of the graph
    # 2. call graph.print_messages to get messages in graphs
    # 3. parsing the messages into vector z 

    # 4. record one step of z 
    # 5. used converged z to get the rate convergence curve


