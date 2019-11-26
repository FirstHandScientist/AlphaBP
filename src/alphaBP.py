import code  # code.interact(local=dict(globals(), **locals()))
import logging
import signal
from factorgraph import Graph
from factorgraph import RV 
from factorgraph import Factor as bpFactor
from factorgraph import divide_safezero
# 3rd party
import numpy as np



logger = logging.getLogger(__name__)

# Settings

# Use this to turn all debugging on or off. Intended use: keep on when you're
# trying stuff out. Once you know stuff works, turn off for speed. Can also
# specify when creating each instance, but this global switch is provided for
# convenience.
DEBUG_DEFAULT = True

# This is the maximum number of iterations that we let loopy belief propagation
# run before cutting it off.
LBP_MAX_ITERS = 50

# Otherwise we'd just make some kinda class to do this anyway.
E_STOP = False


##### classes

class alphaGraph(Graph):
    def __init__(self, alpha=1, debug=DEBUG_DEFAULT, anneal_scheduler=None):
        super(alphaGraph, self).__init__(debug=DEBUG_DEFAULT)
        # add now
        self.debug = debug
        self._rvs = {}

        self._factors = []
        self._alpha = alpha
        self._anneal_scheduler=anneal_scheduler

    # def rv(self, name, n_opts, labels=[], meta={}, debug=DEBUG_DEFAULT):
    #     rv = RV(name, n_opts, labels, meta, debug)
    #     self.add_rv(rv)
    #     return rv


    def factor(self, rvs, name='', potential=None, meta={},
               debug=DEBUG_DEFAULT):
        # Look up RVs if needed.
        for i in range(len(rvs)):
            if debug:
                assert type(rvs[i]) in [str, unicode, RV]
            if type(rvs[i]) in [str, unicode]:
                rvs[i] = self._rvs[rvs[i]]
            # This is just a coding sanity check.
            assert type(rvs[i]) is RV

        f = Factor(rvs, self._alpha, name, potential, meta, debug)
        self.add_factor(f)
        return f

    def lbp(self, init=True, normalize=False, max_iters=LBP_MAX_ITERS,
            progress=False, anneal=False):
        '''
        Loopy belief propagation.
        FAQ:
        -   Q: Do we have do updates in some specific order?
            A: No.
        -   Q: Can we intermix computing messages for Factor and RV nodes?
            A: Yes.
        -   Q: Do we have to make sure we only send messages on an edge once
               messages from all other edges are received?
            A: No. By sorting the nodes, we can kind of approximate this. But
               this constraint is only something that matters if you want to
               converge in 1 iteration on an acyclic graph.
        -   Q: Do factors' potential functions change during (L)BP?
            A: No. Only the messages change.
        '''
        # Sketch of algorithm:
        # -------------------
        # preprocessing:
        # - sort nodes by number of edges
        #
        # note:
        # - every time message sent, normalize if too large or small
        #
        # Algo:
        # - initialize messages to 1
        # - until convergence or max iters reached:
        #     - for each node in sorted list (fewest edges to most):
        #         - compute outgoing messages to neighbors
        #         - check convergence of messages

        nodes = self._sorted_nodes()

        # Init if needed. (Don't if e.g. external func is managing iterations)
        if init:
            self.init_messages(nodes)

        cur_iter, converged = 0, False
        while cur_iter < max_iters and not converged and not E_STOP:
            # Bookkeeping
            

            if progress:
                # self.print_messages(nodes)
                logger.debug('\titeration %d / %d (max)', cur_iter, max_iters)
            
            # ToDo:add the scheduler here for assigning alpha value at each iteration
            if self._anneal_scheduler != None and anneal == True:
                self._alpha = self._anneal_scheduler(global_step=cur_iter)
                # set the factor's _alpha to corresponding value
                self._update_factor_alpha(alpha=self._alpha)
                
            # Comptue outgoing messages:
            converged = True
            for n in nodes:
                n_converged = n.recompute_outgoing(normalize=normalize)
                converged = converged and n_converged
            # increase iteration number 
            cur_iter += 1
        return cur_iter, converged
    
    def _update_factor_alpha(self, alpha):
        '''Update all factors' _alpha value to the given value'''
        for factor in self._factors:
            factor._alpha = alpha
        return self
    

class Factor(bpFactor):
    def __init__(self, rvs, alpha, name='', potential=None, meta={},
                 debug=DEBUG_DEFAULT):
        super(Factor, self).__init__(rvs, name, potential, meta,debug)
        self._alpha = alpha
    

    def recompute_outgoing(self, normalize=False):
        """message need to be computed in max sum version"""
        # Good old safety.
        if self.debug:
            assert self._outgoing is not None, 'must call init_lbp() first'

        # Save old for convergence check.
        old_outgoing = self._outgoing[:]

        # (Product:) Multiply RV messages into "belief".
        incoming = []
        # for alpha type of message, raise power alpha
        belief = np.power(self._potential.copy(), self._alpha)
        for i, rv in enumerate(self._rvs):
            m = rv.get_outgoing_for(self)
            fm = self.get_outgoing_for(rv)

            if self.debug:
                assert m.shape == (rv.n_opts,)
            # Reshape into the correct axis (for combining). For example, if
            # our incoming message (And thus rv.n_opts) has length 3, our
            # belief has 5 dimensions, and this is the 2nd (of 5) dimension(s),
            # then we want the shape of our message to be (1, 3, 1, 1, 1),
            # which means we'll use [1, -1, 1, 1, 1] to project our (3,1) array
            # into the correct dimension.
            #
            # Thanks to stackoverflow:
            # https://stackoverflow.com/questions/30031828/multiply-numpy-
            #     ndarray-with-1d-array-along-a-given-axis
            proj = np.ones(len(belief.shape), int)
            proj[i] = -1
            m_proj = m.reshape(proj)
            fm_proj = np.power(fm.reshape(proj), 1 - self._alpha)
            incoming += [m_proj * fm_proj]
            # Combine to save as we go
            belief *= m_proj * fm_proj

        # Divide out individual belief and (Sum:) add for marginal.
        convg = True
        all_idx = range(len(belief.shape))
        for i, rv in enumerate(self._rvs):
            # get the outgoing message from this fact to rv

            fm = self.get_outgoing_for(rv)
            
            rv_belief = divide_safezero(belief, incoming[i])
            
            axes = tuple(all_idx[:i] + all_idx[i+1:])
            o = rv_belief.sum(axis=axes) * np.power(fm, 1 - self._alpha)
            if self.debug:
                assert self._outgoing[i].shape == (rv.n_opts, )
            if normalize:
                o = divide_safezero(o, sum(o))
            self._outgoing[i] = o
            convg = convg and \
                sum(np.isclose(old_outgoing[i], self._outgoing[i])) == \
                rv.n_opts

        return convg

        
