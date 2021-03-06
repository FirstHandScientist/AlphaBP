import numpy as np
import itertools
import factorgraph as fg
import scipy.sparse.csgraph as csgraph
import maxsum
import alphaBP
import dampedBP
import tree_reweight_BP

import variationalBP
from scipy.stats import multivariate_normal

######################################################################


class MMSE(object):
    def __init__(self, hparam):
        self.constellation = hparam.constellation
        
    def detect(self, y, channel, power_ratio):
        inv = np.linalg.inv(power_ratio * np.eye(channel.shape[1]) 
                            + np.matmul(channel.T, channel) )
        x = inv.dot(channel.T).dot(y)
        
        estimated_x = [self.constellation[np.argmin(np.abs(x_i - np.array(self.constellation)))] for x_i in x]
        return np.array(estimated_x)

class ML(object):
    def __init__(self, hparam):
        self.hparam = hparam
        self.constellation = hparam.constellation
        pass
    
    def detect(self, y, channel, power_ratio):
                
        proposals = list( itertools.product(self.constellation, repeat=channel.shape[1]) )

        threshold = np.inf
        solution = None
        for x in proposals:
            tmp = np.array(channel).dot(x[:]) - y
            if np.dot(tmp, tmp) < threshold:
                threshold = tmp.T.dot(tmp)
                solution = x

        return solution


class EP(object):
    
    def __init__(self, noise_var, hparam):
        #### two intermediate parameters needed to update
        self.gamma = np.zeros(hparam.num_tx*2)
        self.Sigma =  np.ones(hparam.num_tx*2) / hparam.signal_var
        #### two parameters for q(u)
        self.mu = np.zeros_like(self.gamma)
        self.covariance = np.diag(np.ones_like(self.gamma))
        #### two parameters for cavity 
        self.cavity_mgnl_h = np.zeros_like(self.gamma)
        self.cavity_mgnl_t = np.zeros_like(self.gamma)
        #### two prox distribution parameters
        self.prox_mu = np.zeros_like(self.gamma)
        self.prox_var = np.zeros_like(self.gamma)

        self.constellation = hparam.constellation
        

    def get_moments(self):
        return (self.mu, self.covariance)
    

    def update_moments(self, channel, noise_var, noised_signal):
        noised_signal = np.array(noised_signal)
        self.covariance = np.linalg.inv(np.matmul(channel.T, channel)/noise_var
                                        + np.diag(self.Sigma))
        tmp = channel.T.dot(noised_signal)/noise_var
        self.mu = np.dot(self.covariance,
                         tmp+ self.gamma )
        #assert np.all(self.covariance>=0)

    def update_cavity(self):
        for i in range(self.mu.shape[0]):
            self.cavity_mgnl_h[i] = self.covariance[i, i]/( 1 - self.covariance[i, i] * self.Sigma[i])
            self.cavity_mgnl_t[i] = self.cavity_mgnl_h[i] * (self.mu[i]/self.covariance[i,i] - self.gamma[i])
            assert np.all(self.cavity_mgnl_h>=0)


    def update_prox_moments(self):
        vary_small = 1e-6
        #gaussian = GaussianDiag()
        mean = self.cavity_mgnl_t
        #logs = np.log(self.cavity_mgnl_h)/2
        z = None
        for i, the_mean in enumerate(mean):
            # logp = gaussian.likelihood(mean=the_mean,
            #                            logs=logs[i],
            #                            x= np.array(self.constellation))
            logp = np.log(multivariate_normal.pdf(x=np.array(self.constellation),
                                                  mean=the_mean,
                                                  cov=self.cavity_mgnl_h[i]) + vary_small )
            z = np.sum(np.exp(logp))
            self.prox_mu[i] = np.array(self.constellation).dot( np.exp(logp) )/z
            # second_moment = np.power(np.array(self.constellation), 2).dot( np.exp(logp) )/z
            # self.prox_var[i] = second_moment -  np.power(self.prox_mu[i], 2)
            self.prox_var[i] = np.power(np.array(self.constellation) - self.prox_mu[i], 2).dot( np.exp(logp) )/z
            assert np.all(self.prox_var>=0)

    def kl_match_momoents(self):
        vary_small = 1e-6
        Sigma = 1./(self.prox_var + vary_small) - 1./(self.cavity_mgnl_h + vary_small)
        gamma = self.prox_mu / (self.prox_var+vary_small) - self.cavity_mgnl_t / (self.cavity_mgnl_h+vary_small)
        if np.any(np.isnan(Sigma)) and np.any(np.isnan(gamma)):
            print("value error")
        if np.any(Sigma>0):
            positive_idx = Sigma>0
            self.Sigma[positive_idx] = Sigma[positive_idx]
            self.gamma[positive_idx] = gamma[positive_idx]
            assert np.all(self.Sigma>0)

    def fit(self, channel, noise_var, noised_signal, stop_iter=10):
        """Do the training by number of iteration of stop_iter"""
        for i in range(stop_iter):
            self.update_moments(channel=channel,
                                noise_var=noise_var,
                                noised_signal=noised_signal)
            self.update_cavity()
            self.update_prox_moments()
            self.kl_match_momoents()


    def detect_signal_by_mean(self):
        estimated_signal = []
        for mu in self.mu:
            obj_list  = np.abs(mu - np.array(self.constellation))
            estimated_signal.append(self.constellation[np.argmin(obj_list)])
        return estimated_signal

    
    def detect_signal_by_map(self):
        
        mean = self.mu
        cov = self.covariance
        proposals = list( itertools.product(self.constellation, repeat=mean.shape[0]) )
        
        p_list = multivariate_normal.pdf(x=proposals, mean=mean, cov=cov) 

        # for x in proposals:
        #     logp = np.log(multivariate_normal.pdf(x=x, mean=mean, cov=cov) )
        #     logp_list.append(logp)
        idx_max = np.argmax( p_list )
        return proposals[idx_max]

class NaiveMF(object):
    """
    The naive mean field method.
    """
    def __init__(self, noise_var, hparam):
        # get the constellation
        self.constellation = hparam.constellation
        self.hparam = hparam
        self.n_symbol = hparam.num_tx * 2 # the number of vairiables


    def _init_potential(self, h_matrix, observation, noise_var):
        # 1 * n  
        self.unary = np.matmul(h_matrix.T, observation) / noise_var
        # n * n
        self.binary =  - 0.5 * np.matmul(h_matrix.T, h_matrix) / noise_var
        

    def mf_update(self, num_iters = 1, mean = None):
        # mean : n ** 2 of mean-field marginals
        binary = self.binary
        unary = self.unary

        if mean is None:
            mean = np.ones_like(self.unary) * 0.5
        for _ in range(num_iters):
            for n in np.random.permutation(range(self.n_symbol)):
                message = 0
                for k in range(self.n_symbol):
                    # fully connected graph, rest nodes all are neighbors
                    binary_nk = binary[n][k]
                    binary_nk = binary_nk
                    mean_k = mean[k]*2-1
                    message += mean_k*binary_nk
                    
                message += unary[n]
                # sigmoid
                mean[n] = 1 / ( 1 + np.exp(-2*message))
                
        return mean

    def fit(self, channel, noise_var, noised_signal, stop_iter=100):
        """ set potentials and run message passing"""
        self._init_potential(h_matrix=channel,
                             observation=noised_signal,
                             noise_var=noise_var)
        unary_marginals_mf = np.ones(self.n_symbol) * 0.5
        for i in range(stop_iter):
            unary_marginals_mf_new = self.mf_update(1, unary_marginals_mf.copy())
            if np.linalg.norm(unary_marginals_mf-unary_marginals_mf_new) < 1e-6:
                break

            unary_marginals_mf = unary_marginals_mf_new.copy()

        
        self.mu = unary_marginals_mf_new
        # run BP

        
    def detect_signal_by_mean(self):
        estimated_signal = []
        for mu in self.mu:
            if mu >= 0.5:
                estimated_signal.append(self.constellation[1])
            else:
                estimated_signal.append(self.constellation[0])
            
        return estimated_signal

    

class LoopyBP(object):

    def __init__(self, noise_var, hparam):
        # get the constellation
        self.constellation = hparam.constellation
        self.hparam = hparam
        # set the graph
        self.graph = fg.Graph()
        # add the discrete random variables to graph
        self.n_symbol = hparam.num_tx * 2
        for idx in range(hparam.num_tx * 2):
            self.graph.rv("x{}".format(idx), len(self.constellation))

    def set_potential(self, h_matrix, observation, noise_var):
        s = np.matmul(h_matrix.T, h_matrix)
        for var_idx in range(h_matrix.shape[1]):
            # set the first type of potentials, the standalone potentials
            f_potential = (-0.5 *s[var_idx, var_idx] * np.power(self.constellation, 2) + h_matrix[:, var_idx].dot(observation) * np.array(self.constellation))/noise_var
            
            f_x_i = np.exp(f_potential )
            self.graph.factor(["x{}".format(var_idx)],
                              potential=f_x_i)

        for var_idx in range(h_matrix.shape[1]):

            for var_jdx in range(var_idx + 1, h_matrix.shape[1]):
                # set the cross potentials
                t_potential = - np.array(self.constellation)[None,:].T * s[var_idx, var_jdx] * np.array(self.constellation) / noise_var
                
                t_ij = np.exp(t_potential)
                self.graph.factor(["x{}".format(var_jdx), "x{}".format(var_idx)],
                                  potential=t_ij)
        
    
    def fit(self, channel, noise_var, noised_signal, stop_iter=100):
        """ set potentials and run message passing"""
        self.set_potential(h_matrix=channel,
                           observation=noised_signal,
                           noise_var=noise_var)

        # run BP
        iters, converged = self.graph.lbp(normalize=True, max_iters=stop_iter)
        
    def detect_signal_by_mean(self):
        estimated_signal = []
        rv_marginals = dict(self.graph.rv_marginals())
        for idx in range(self.n_symbol):
            x_marginal = rv_marginals["x{}".format(idx)]
            
            estimated_signal.append(self.constellation[x_marginal.argmax()])
        return estimated_signal
    
class AlphaBP(LoopyBP):
    def __init__(self, noise_var, hparam):
        self.hparam = hparam
        # get the constellation
        self.constellation = hparam.constellation

        self.n_symbol = hparam.num_tx * 2
        # set the graph
        self.graph = alphaBP.alphaGraph(alpha=hparam.alpha)
        # add the discrete random variables to graph
        for idx in range(hparam.num_tx * 2):
            self.graph.rv("x{}".format(idx), len(self.constellation))

class DampBP(LoopyBP):
    def __init__(self, noise_var, hparam):
        self.hparam = hparam
        # get the constellation
        self.constellation = hparam.constellation

        self.n_symbol = hparam.num_tx * 2
        # set the graph
        self.graph = dampedBP.dampedGraph(eta=hparam.eta)
        # add the discrete random variables to graph
        for idx in range(hparam.num_tx * 2):
            self.graph.rv("x{}".format(idx), len(self.constellation))


class AnnealAlphaBP(AlphaBP):
    """Annealing alpha value in iterations of alphaBP"""
    def __init__(self, noise_var, hparam):
        self.hparam = hparam
        # get the constellation
        self.constellation = hparam.constellation

        self.n_symbol = hparam.num_tx * 2
        # set the graph
        self.graph = alphaBP.alphaGraph(alpha=hparam.alpha,
                                        anneal_scheduler=hparam.alpha_schedule)

        # add the discrete random variables to graph
        for idx in range(hparam.num_tx * 2):
            self.graph.rv("x{}".format(idx), len(self.constellation))




    def fit(self, channel, noise_var, noised_signal, stop_iter=10):
        """ set potentials and run message passing"""
        self.set_potential(h_matrix=channel,
                           observation=noised_signal,
                           noise_var=noise_var)

        # run BP with anneal as true
        iters, converged = self.graph.lbp(normalize=True, max_iters=stop_iter, anneal=True)

class TreeReweightBP(LoopyBP):
    def __init__(self, noise_var, hparam):
        self.hparam = hparam
        # get the constellation
        self.constellation = hparam.constellation
        # For complete graph, the appear rate of each edge is a constant
        self.weight = hparam.TBP_rate 
        self.n_symbol = hparam.num_tx * 2
        # set the graph
        self.graph = tree_reweight_BP.tree_reweight_Graph(weight=self.weight)

        self.unary_factors = []
        # add the discrete random variables to graph
        for idx in range(hparam.num_tx * 2):
            self.graph.rv("x{}".format(idx), len(self.constellation))



    def set_potential(self, h_matrix, observation, noise_var):
        s = np.matmul(h_matrix.T, h_matrix)
        for var_idx in range(h_matrix.shape[1]):
            # set the first type of potentials, the standalone potentials
            f_potential = (-0.5 *s[var_idx, var_idx] * np.power(self.constellation, 2) + h_matrix[:, var_idx].dot(observation) * np.array(self.constellation))/noise_var
            
            f_x_i = np.exp(f_potential )
            self.unary_factors.append(f_x_i)
            self.graph.factor(["x{}".format(var_idx)], potential=f_x_i)

        for var_idx in range(h_matrix.shape[1]):

            for var_jdx in range(var_idx + 1, h_matrix.shape[1]):
                # set the cross potentials
                t_potential = - np.array(self.constellation)[None,:].T * s[var_idx, var_jdx] * np.array(self.constellation) / noise_var
                
                t_ij = np.exp(t_potential)
                self.graph.factor(["x{}".format(var_jdx), "x{}".format(var_idx)],
                                  potential=t_ij)

    def detect_signal_by_mean(self):
        estimated_signal = []
        rv_marginals = dict(self.graph.rv_marginals())
        for idx in range(self.n_symbol):
            x_marginal = rv_marginals["x{}".format(idx)]
            unary_potential = self.unary_factors[idx].copy()
            unary_potential = unary_potential / unary_potential.sum()
            beliefs = x_marginal / unary_potential
            x_marginal = np.power(beliefs, self.weight) * unary_potential
            estimated_signal.append(self.constellation[x_marginal.argmax()])

        return estimated_signal

    


class StochasticBP(AlphaBP):
    def __init__(self, noise_var, hparam):
        self.hparam = hparam
        # get the constellation
        self.constellation = hparam.constellation
        self.alpha = hparam.alpha
        self.n_symbol = hparam.num_tx * 2
        # set the graph
        self.learning_rate = 1
        self.first_iter_flag = True


    def subgraph_mask(self, size):
        """give the mask for spanning tree subgraph"""
        init_matrix = np.random.randn(size,size)
        Tcs = csgraph.minimum_spanning_tree(init_matrix)
        mask_matrix = Tcs.toarray()
        return mask_matrix

    def new_graph(self, h_matrix, observation, noise_var):
        # initialize new graph
        subgraph = alphaBP.alphaGraph(alpha=self.alpha)
        # add the discrete random variables to graph
        for idx in range(h_matrix.shape[1]):
            subgraph.rv("x{}".format(idx), len(self.constellation))

        s = np.matmul(h_matrix.T, h_matrix)

        # get the prior belief
        if not self.first_iter_flag:
            rv_marginals = dict(self.graph.rv_marginals())

        for var_idx in range(h_matrix.shape[1]):
            # set the first type of potentials, the standalone potentials
            f_potential = (-0.5 *s[var_idx, var_idx] * np.power(self.constellation, 2)
                             + h_matrix[:, var_idx].dot(observation) * np.array(self.constellation))/noise_var
            
            f_x_i = np.exp( f_potential)
            f_x_i = f_x_i/f_x_i.sum()
            if not self.first_iter_flag:
                old_prior = rv_marginals["x{}".format(var_idx)]
                subgraph.factor(["x{}".format(var_idx)],
                                potential=np.power(f_x_i, self.learning_rate) * old_prior)
            else:
                subgraph.factor(["x{}".format(var_idx)],
                                potential=f_x_i)
        ## sampling the subgraph mask first and set cross potentials
        graph_mask = self.subgraph_mask(h_matrix.shape[1])
        
        for var_idx in range(h_matrix.shape[1]):
            
            for var_jdx in range(var_idx + 1, h_matrix.shape[1]):
                # set the cross potentials
                test_condition = np.isclose(np.array([graph_mask[var_idx, var_jdx],
                                                      graph_mask[var_jdx, var_idx]]),
                                            np.array([0,0]))
                
                if not np.all(test_condition):
                    t_potential = - np.array(self.constellation)[None,:].T * s[var_idx, var_jdx] * np.array(self.constellation) / noise_var
                    
                    t_ij = np.exp(t_potential)
                    t_ij = t_ij/t_ij.sum()
                    subgraph.factor(["x{}".format(var_jdx), "x{}".format(var_idx)],
                                    potential= np.power(t_ij, self.learning_rate))
        return subgraph
        

    
    def fit(self, channel, noise_var, noised_signal, stop_iter=10):
        rate_list = np.linspace(1, 0.01, stop_iter)
        for iti in range(stop_iter):
            # initialize a new graph
            self.learning_rate = rate_list[iti]
            """ set potentials and run message passing"""
            self.graph = self.new_graph(h_matrix=channel,
                                        observation=noised_signal,
                                        noise_var=noise_var)

            # run BP
            iters, converged = self.graph.lbp(normalize=True,
                                              max_iters=50)

            self.first_iter_flag = False


    
    

class VariationalBP(LoopyBP):
    def __init__(self, noise_var, hparam):
        self.hparam = hparam
        # get the constellation
        self.constellation = hparam.constellation

        self.n_symbol = hparam.num_tx * 2
        # set the graph
        self.graph = variationalBP.variationalGraph()
        # add the discrete random variables to graph
        for idx in range(hparam.num_tx * 2):
            self.graph.rv("x{}".format(idx), len(self.constellation))


class MMSEalphaBP(AlphaBP):
    def set_potential(self, h_matrix, observation, noise_var):
        power_ratio = noise_var/self.hparam.signal_var
        s = np.matmul(h_matrix.T, h_matrix)
        inv = np.linalg.inv(power_ratio * np.eye(h_matrix.shape[1]) 
                            + s )
        prior_u = inv.dot(h_matrix.T).dot(observation)
                        
        for var_idx in range(h_matrix.shape[1]):
            # set the first type of potentials, the standalone potentials
            f_potential = (-0.5 *s[var_idx, var_idx] * np.power(self.constellation, 2)
                             + h_matrix[:, var_idx].dot(observation) * np.array(self.constellation))/noise_var
            
            f_x_i = np.exp( f_potential )
            
            p_potential = -0.5 * np.power(self.constellation - prior_u[var_idx], 2) \
                             / (inv[var_idx, var_idx] * noise_var) 
            
            prior_i = np.exp(p_potential)
            self.graph.factor(["x{}".format(var_idx)],
                              potential=f_x_i * prior_i)


        for var_idx in range(h_matrix.shape[1]):

            for var_jdx in range(var_idx + 1, h_matrix.shape[1]):
                # set the cross potentials
                t_potential = - np.array(self.constellation)[None,:].T * s[var_idx, var_jdx] * np.array(self.constellation) / noise_var
                
                t_ij = np.exp(t_potential)
                self.graph.factor(["x{}".format(var_jdx), "x{}".format(var_idx)],
                                  potential=t_ij)


    
class MMSEvarBP(VariationalBP):
    def set_potential(self, h_matrix, observation, noise_var):
        power_ratio = noise_var/self.hparam.signal_var
        s = np.matmul(h_matrix.T, h_matrix)
        inv = np.linalg.inv(power_ratio * np.eye(h_matrix.shape[1]) 
                            + s )
        prior_u = inv.dot(h_matrix.T).dot(observation)
                        
        for var_idx in range(h_matrix.shape[1]):
            # set the first type of potentials, the standalone potentials
            f_x_i = np.exp( (-0.5 *s[var_idx, var_idx] * np.power(self.constellation, 2)
                             + h_matrix[:, var_idx].dot(observation) * np.array(self.constellation))/noise_var)
            prior_i = np.exp(-0.5 * np.power(self.constellation - prior_u[var_idx], 2) \
                             / (inv[var_idx, var_idx] * noise_var))
            self.graph.factor(["x{}".format(var_idx)],
                              potential=f_x_i * prior_i)


        for var_idx in range(h_matrix.shape[1]):

            for var_jdx in range(var_idx + 1, h_matrix.shape[1]):
                # set the cross potentials
                t_ij = np.exp(- np.array(self.constellation)[None,:].T
                              * s[var_idx, var_jdx] * np.array(self.constellation) / noise_var)
                self.graph.factor(["x{}".format(var_jdx), "x{}".format(var_idx)],
                                  potential=t_ij)
    
