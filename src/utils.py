import numpy as np
import igraph

def channel_component(hparam):
    return 1/np.sqrt(2*hparam.num_rx)*np.random.randn(hparam.num_rx,hparam.num_tx)

def sampling_signal(hparam):
    x = np.random.choice(hparam.constellation, hparam.num_tx*2, replace=True, p=hparam.soucrce_prior)
    symbol = real2complex(x)
    return x, symbol

def sampling_H(hparam):
    real = channel_component(hparam)
    img = channel_component(hparam)
    real_img = np.concatenate((real, -img), axis=1)
    img_real = np.concatenate((img, real), axis=1)
    return np.concatenate((real_img, img_real), axis=0)

def ERsampling_S(hparam, p):
    S = np.zeros((hparam.num_tx, hparam.num_rx))
    b = np.random.randn(hparam.num_tx)* hparam.stn_var / 4
    
    for i in range(hparam.num_tx):
        
        for j in range(i, hparam.num_rx):
            trial = np.random.random()
            if trial <= p:
                S[i, j] = np.random.randn() * hparam.stn_var
                S[j, i] = S[i, j]
        
        S[i, i] = np.abs(np.random.randn())
        # row_max = S[i].max()
        # if S[i, i]< row_max:
        #     S[i, i] = row_max
    
    return (S, b)
    
def sampling_noise(hparam, snr):
    # noise_var = hparam.num_tx/hparam.num_rx * np.power(10, -snr/10)
    # noise_var = hparam.num_tx * np.power(10, -snr/10)
    noise_var = 1. / snr
    noise = np.sqrt( noise_var) * np.random.randn(hparam.num_rx * 2)
    return (noise, noise_var)

                        
def real2complex(x):
    x = np.array(x)
    num = x.shape[0]
    real = x[:int(num/2)]
    img = x[int(num/2):num]
    return real + 1j * img

## convergence condition of theorem 1
def largest_singular(S, alpha):
    np.fill_diagonal(S, 0) 
    aj_matrix = 2 * S
    #create the graph by aj_matrix
    graph = igraph.Graph.Adjacency(aj_matrix.astype(bool).tolist())
    # get all directed edges in graph
    edges = graph.get_edgelist()
    # construct the m_matrix for convergence check
    m_matrix = np.zeros((len(edges), len(edges)))
    for i, row in enumerate(edges):
        for j, col in enumerate(edges):
            if row == col:
                m_matrix[i,j] = np.abs(1 - alpha)
            elif row == col[::-1]:
                m_matrix[i,j] = np.abs(1 - alpha) * np.tanh( np.abs( alpha * aj_matrix[row] ))

            elif row[0] == col[1]:
                if row[1] == col[0]:
                    pass
                else:
                    m_matrix[i,j] = np.tanh( np.abs( alpha * aj_matrix[row] ))
    
    _, singulars, _ = np.linalg.svd(m_matrix)
    return singulars.max()

def converge_cond_m(S, alpha):
    # does S contain potentials???
    # set the diagonal entry to zero since converge condition does not need it
    max_singular = largest_singular(S, alpha)
    cnvg = True if max_singular < 1 else False
    return cnvg



### message operation functions

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

