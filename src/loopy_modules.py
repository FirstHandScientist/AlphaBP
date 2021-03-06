import numpy as np
import itertools
import factorgraph as fg
import maxsum
import alphaBP
from scipy.stats import multivariate_normal


######################################################################


class ML(object):
    def __init__(self, hparam):
        self.hparam = hparam
        self.constellation = hparam.constellation
        pass
    
    def detect(self, S, b):
                
        proposals = list( itertools.product(self.constellation, repeat=self.hparam.num_tx) )

        threshold = np.inf
        solution = None
        for x in proposals:
            tmp = np.matmul(np.array(x), S).dot(np.array(x)) + b.dot(x)
            if tmp < threshold:
                threshold = tmp
                solution = x

        return solution

class Marginal(object):
    """Compute all the marginals for a given distributions"""
    def __init__(self, hparam):
        self.hparam = hparam
        self.constellation = hparam.constellation
        pass
    
    def detect(self, S, b):
        
        proposals = list( itertools.product(self.constellation, repeat=S.shape[0]) )
        array_proposals = np.array(proposals)
        prob = []
        for x in proposals:
            tmp = np.matmul(np.array(x), S).dot(np.array(x)) + b.dot(x)
            prob.append(np.exp(-tmp))
            
        prob = np.array(prob)
        
        marginals = []
        for i in range(b.shape[0]):
            this_marginal = []
            for code in self.constellation:
                subset_idx = array_proposals[:, i]==code
                this_marginal.append(np.sum( prob[subset_idx]))
            
            # normalize the marginal
            this_marginal = np.array(this_marginal)
            this_marginal = this_marginal/this_marginal.sum()
            
            marginals.append( this_marginal)
        
        return np.array(marginals)


class LoopyBP(object):

    def __init__(self, noise_var, hparam):
        # get the constellation
        self.constellation = hparam.constellation
        self.hparam = hparam
        # set the graph
        self.graph = fg.Graph()
        # add the discrete random variables to graph
        self.n_symbol = hparam.num_tx 
        for idx in range(hparam.num_tx):
            self.graph.rv("x{}".format(idx), len(self.constellation))

    def set_potential(self, S, b):
        s = S
        for var_idx in range(self.hparam.num_tx):
            # set the first type of potentials, the standalone potentials
            f_x_i = np.exp( - s[var_idx, var_idx] * np.power(self.constellation, 2)
                             - b[var_idx] * np.array(self.constellation))
            self.graph.factor(["x{}".format(var_idx)],
                              potential=f_x_i)


        for var_idx in range(self.hparam.num_tx):

            for var_jdx in range(var_idx + 1, self.hparam.num_tx):
                # set the cross potentials
                if s[var_idx, var_jdx] > 0:
                    
                    t_ij = np.exp(-2* np.array(self.constellation)[None,:].T
                                  * s[var_idx, var_jdx] * np.array(self.constellation))
                    self.graph.factor(["x{}".format(var_jdx), "x{}".format(var_idx)],
                                      potential=t_ij)


    
    def fit(self, S, b, stop_iter=10):
        """ set potentials and run message passing"""
        self.set_potential(S, b)

        # run BP
        iters, converged = self.graph.lbp(normalize=True,max_iters=stop_iter)

        
    def lbp_iteration(self, check_skip=5, stop_iter=50):
        '''
        do iterations by the given number of check_skip
        then return the messages in the graph
        '''
        for i in range(check_skip):
            self.graph.lbp_iteration(normalize=True, max_iters=stop_iter)
        all_sorted_messages = self.graph.get_messages()
        
        return all_sorted_messages

        
    def detect_signal_by_mean(self):
        estimated_signal = []
        rv_marginals = dict(self.graph.rv_marginals())
        for idx in range(self.n_symbol):
            x_marginal = rv_marginals["x{}".format(idx)]
            
            estimated_signal.append(self.constellation[x_marginal.argmax()])
        return estimated_signal
    
    def marginals(self):
        marginal_prob = []
        rv_marginals = dict(self.graph.rv_marginals())
        for idx in range(self.n_symbol):
            x_marginal = rv_marginals["x{}".format(idx)]
            x_marginal = np.array(x_marginal)
            x_marginal = x_marginal/x_marginal.sum()
            marginal_prob.append(x_marginal)
        return np.array(marginal_prob)

    

    
class AlphaBP(LoopyBP):
    def __init__(self, noise_var, hparam):
        self.hparam = hparam
        # get the constellation
        self.constellation = hparam.constellation

        self.n_symbol = hparam.num_tx 
        # set the graph
        self.graph = alphaBP.alphaGraph(alpha=hparam.alpha)
        # add the discrete random variables to graph
        for idx in range(hparam.num_tx ):
            self.graph.rv("x{}".format(idx), len(self.constellation))

class MMSEalphaBP(AlphaBP):
    def set_potential(self, S, b):
        
        s = S 
        inv = np.linalg.inv(np.eye(s.shape[0]) + 2 * s )
        prior_u = inv.dot(b)
                        
        for var_idx in range(s.shape[1]):
            # set the first type of potentials, the standalone potentials
            f_x_i = np.exp( - s[var_idx, var_idx] * np.power(self.constellation, 2)
                             - b[var_idx] * np.array(self.constellation))
            
            prior_i = np.exp(-0.5 * np.power(self.constellation - prior_u[var_idx], 2) \
                             / (inv[var_idx, var_idx]) )
            self.graph.factor(["x{}".format(var_idx)],
                              potential=f_x_i * prior_i)


        for var_idx in range(s.shape[1]):

            for var_jdx in range(var_idx + 1, s.shape[1]):
                # set the cross potentials
                if s[var_idx, var_jdx] > 0:
                    
                    t_ij = np.exp(- 2 * np.array(self.constellation)[None,:].T
                                  * s[var_idx, var_jdx] * np.array(self.constellation))
                    self.graph.factor(["x{}".format(var_jdx), "x{}".format(var_idx)],
                                      potential=t_ij)
