import numpy as np
states = A, B, C = 0, 1, 2

#起始機率pi 
pi = np.array([0.33, 0.33, 0.34])
#狀態傳輸機率A
pA = np.array([[0.34, 0.33, 0.33], [0.33, 0.34, 0.33], [0.33, 0.33, 0.34]])
#發射（混合）機率O
O = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

def forward(params, observations):
    pi, pA, O = params
    N = len(observations)
    S = pi.shape[0]
    
    alpha = np.zeros((N, S))
    
    # base case
    alpha[0, :] = pi * O[:,observations[0]]
    
    # recursive case
    for i in range(1, N):
        for s2 in range(S):
            for s1 in range(S):
                alpha[i, s2] += alpha[i-1, s1] * pA[s1, s2] * O[s2, observations[i]]
    
    return (alpha, np.sum(alpha[N-1,:]))
    #return (np.sum(alpha[N-1,:]))

def backward(params, observations):
    pi, pA, O = params
    N = len(observations)
    S = pi.shape[0]
    
    beta = np.zeros((N, S))
    
    # base case
    beta[N-1, :] = 1
    
    # recursive case
    for i in range(N-2, -1, -1):
        for s1 in range(S):
            for s2 in range(S):
                beta[i, s1] += beta[i+1, s2] * pA[s1, s2] * O[s2, observations[i+1]]
    
    return (beta, np.sum(pi * O[:, observations[0]] * beta[0,:]))


def baum_welch(training, pi, pA, O, iterations):
    pi, pA, O = np.copy(pi), np.copy(pA), np.copy(O) # take copies, as we modify them
    S = pi.shape[0]
    
    # do several steps of EM hill climbing
    for it in range(iterations):
        pi1 = np.zeros_like(pi)
        A1 = np.zeros_like(pA)
        O1 = np.zeros_like(O)
        
        for observations in training:
            # compute forward-backward matrices
            alpha, za = forward((pi, pA, O), observations)
            beta, zb = backward((pi, pA, O), observations)
            #assert abs(za - zb) < 1e-6, "it's badness 10000 if the marginals don't agree"
            
            # M-step here, calculating the frequency of starting state, transitions and (state, obs) pairs
            pi1 += alpha[0,:] * beta[0,:] / za
            for i in range(0, len(observations)):
                O1[:, observations[i]] += alpha[i,:] * beta[i,:] / za
            for i in range(1, len(observations)):
                for s1 in range(S):
                    for s2 in range(S):
                        A1[s1, s2] += alpha[i-1,s1] * pA[s1, s2] * O[s2, observations[i]] * beta[i,s2] / za
                                                                    
        # normalise pi1, A1, O1
        pi = pi1 / np.sum(pi1)
        for s in range(S):
            pA[s, :] = A1[s, :] / np.sum(A1[s, :])
            O[s, :] = O1[s, :] / np.sum(O1[s, :])
        print("pi",pi)
        print("A",pA)
        print("O",O)
    return pi, pA, O

dataset1 = [[A,B,B,C,A,B,C,A,A,B,C ],
            [A,B,C,A,B,C],
            [A,B,C,A,A,B,C],
            [B,B,A,B,C,A,B],
            [B,C,A,A,B,C,C,A,B],
            [C,A,C,C,A,B,C,A],
            [C,A,B,C,A,B,C,A],
            [C,A,B,C,A],
            [C,A,B,C,A ]]
dataset2 = [[B,B,B,C,C,B,C],
            [C,C,B,A,B,B],
            [A,A,C,C,B,B,B ],
            [B,B,A,B,B,A,C ],
            [C,C,A,A,B,B,A,B],
            [B,B,B,C,C,B,A,A],
            [A,B,B,B,B,A,B,A],
            [C,C,C,C,C],
            [B,B,A,A,A,]]

print("dataset 1")            
pi2, A2, O2 = baum_welch(dataset1, pi, pA, O, 50)
print("dataset 2")    
pi3, A3, O3 = baum_welch(dataset2, pi, pA, O, 50)


for data in dataset1 + dataset2 + [[A,B,C,A,B,C,C,A,B], [A,A,B,A,B,C,C,C,C,B,B,B]]:
    h1,sumh1 = forward((pi2, A2, O2), data)
    h2 ,sumh2= forward((pi3, A3, O3), data)
   
    #print(data, end = ' ')
    if(sumh1 > sumh2):
        print(data,'hmm1')
    else:
        print(data,'hmm2')

