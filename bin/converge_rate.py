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


# import the methods and sources
sys.path.append("./src")
from loopy_modules import LoopyBP, AlphaBP, ML
from utils import channel_component, sampling_noise, sampling_signal, sampling_H,real2complex, ERsampling_S



# configuration
class hparam(object):
    num_tx = 8
    num_rx = 8
    soucrce_prior = [0.5, 0.5]
    signal_var = 1
    stn_var= 1
    connect_prob = np.linspace(0.0, 0.9, 10)
    monte = 20
    power_n = 4./3
    constellation = [int(-1), int(1)]

    alpha = None
    # algos = {"AlphaBP, 0.5": {"detector": AlphaBP, "alpha": 0.5},
    #          "AlphaBP, 1": {"detector": AlphaBP, "alpha": 1},
    #          "AlphaBP, 1.2": {"detector": AlphaBP, "alpha": 1.2}
    # }
    algos = {"AlphaBP, 0.5": {"detector": AlphaBP, "alpha": 0.5, "legend": r'$\alpha=$,'+' {}'.format(0.5)}
    }
    
    # total number of iterations
    iter_num = 100
    # the number of iterations before each each checkpoint
    check_skip = 5
    
    for _, value in algos.items():
        value["ratio"] = []

def list_message_to_norm(messages):
    '''compute the norm2 of a given list of list of messages'''
    tmp_sum = 0
    for nodes in messages:
        for n in nodes:
            tmp_sum += np.power(n, 2).sum()
    
    return np.power(tmp_sum, 0.5)

def list_message_diff(messages1, messages2):
    '''
    return the difference between two list of messages
    '''
    messages_diff = [ [n - messages2[i][j] for j, n in enumerate(nodes)] for i, nodes in enumerate(messages1)]
    return messages_diff

def messages_to_norm_ratio(sorted_messages, pop_last=True):
    '''
    input: a collection of checkpoints of messages 
    output: the log ratio of norm2 compared to the last message set
    '''
    conveged_messages = sorted_messages[-1]
    
    mssg_ratio = []
    for messages_step_n in sorted_messages:
        mssg_diff = list_message_diff(messages_step_n, conveged_messages)
        mssg_ratio.append( list_message_to_norm(mssg_diff)/
                          list_message_to_norm(conveged_messages))
    # last diff is 0, pop it out
    mssg_ratio.pop()
    return np.array(mssg_ratio)


    

if __name__ == "__main__":
    usage = "python bin/converge_rate.py"

    erp = 0.4

    tmp = dict()
    for name,_ in hparam.algos.items():
        tmp[name] = []
    for monte in range(hparam.monte):

        # sampling the S and b for exponential function
        S, b = ERsampling_S(hparam, erp)

        # compute the joint ML detection
        # detectML = ML(hparam)
        # solution = detectML.detect(S, b)
        
        for key, method in hparam.algos.items():
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
        ax.semilogy(range(0, hparam.iter_num - hparam.check_skip, hparam.check_skip),
                    np.array(method["ratio"]).mean(axis=0),
                    marker=next(iter_marker_list),
                    label=method['legend'])
        ax.fill_between(range(0, hparam.iter_num - hparam.check_skip, hparam.check_skip),
                        np.array(method["ratio"]).min(axis=0),
                        np.array(method["ratio"]).max(axis=0))
    
    ax.set(xlabel="Iteration", ylabel="log ratio")
    # lgd = ax.legend(bbox_to_anchor=(1.64,1), borderaxespad=0)
    ax.grid()
    plt.show()
    # 1. done one step of lbp of the graph
    # 2. call graph.print_messages to get messages in graphs
    # 3. parsing the messages into vector z 

    # 4. record one step of z 
    # 5. used converged z to get the rate convergence curve


