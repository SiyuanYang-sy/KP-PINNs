import numpy as np
from scipy.sparse import spdiags


# A, Phi = compute_APhi(X, 1, 1/2)
# A, Phi = compute_APhi(X, 1, 3/2)
# A, Phi = compute_APhi(X, 1, 5/2)

def KP_compute_APhi(X, theta, nu):  # theta = 1, nu = 1/2, 3/2, 5/2
    if nu == 1/2:
        n = X.shape[0]
        X = X * theta
        vec1 = X[1:] - X[:-1]
        upper_diag = 1 / (np.exp(vec1) - np.exp(-vec1))
        
        vec2 = X[2:] - X[:-2]
        diag_m = (np.exp(vec2) - np.exp(-vec2)) * upper_diag[1:] * upper_diag[:-1]
        
        diag_m = np.insert(diag_m, 0, np.exp(X[1] - X[0]) * upper_diag[0]) 
        diag_m = np.insert(diag_m, len(diag_m), np.exp(X[-1] - X[-2]) * upper_diag[-1]) 
        
        
        K = np.array([np.exp(-abs(X[0] - X[0])),
                      np.exp(-abs(X[0] - X[1]))])  
        A_l_1 = np.array([[diag_m[0]],
                          [upper_diag[0]]])  
        #Phi_l_1 = K @ A_l_1   
        
        Phi_central = np.zeros(n - 2)
        for i in range(2, n):
            K = np.array([np.exp(-abs(X[i - 1] - X[i - 2])),
                          np.exp(-abs(X[i - 1] - X[i - 1])),
                          np.exp(-abs(X[i - 1] - X[i]))])  
            A_central = np.array([[upper_diag[i - 2]],
                                  [diag_m[i - 1]],
                                  [upper_diag[i - 1]]])  
            Phi_central[i - 2] = K @ A_central   
        
        K = np.array([np.exp(-abs(X[n - 1] - X[n - 2])),
                      np.exp(-abs(X[n - 1] - X[n - 1]))])  
        A_r_1 = np.array([[upper_diag[-1]],
                          [diag_m[-1]]])  
        #Phi_r_1 = K @ A_r_1   
        
        
        A = spdiags(diag_m, 0, n, n)
        A = A + spdiags(-upper_diag, -1, n, n)
        A = A + spdiags(-upper_diag, -1, n, n).transpose()
        
        A_csr = A.tocsr()  
        
        '''P = Phi_central
        P = np.insert(P, 0, Phi_l_1)
        P = np.insert(P, len(P), Phi_r_1)
        
        Phi = spdiags(P, 0, n, n)'''

        # Creates the unit diagonal matrix as Phi (when nu = 1/2, A: inverse of the kernel covaiance matrix K)
        Phi_data = np.ones(n)
        Phi = spdiags(Phi_data, 0, n, n)
        
        Phi_csr = Phi.tocsr()  

        return A_csr, Phi_csr
    
    elif nu == 3/2:
        n = X.shape[0]
        X = theta * X
        h = 1 / n ** 2
        
        S = np.vstack((np.exp(X[0:3]), X[0:3] * np.exp(X[0:3]))).T
        Q, _ = np.linalg.qr(S, mode='complete')  
        A_l_1 = Q[:, -1] / h
        K = np.array([[((1 + abs(X[0] - X[0])) * np.exp(-abs(X[0] - X[0]))),
                       ((1 + abs(X[0] - X[1])) * np.exp(-abs(X[0] - X[1]))),
                       ((1 + abs(X[0] - X[2])) * np.exp(-abs(X[2] - X[0])))],
                      [((1 + abs(X[1] - X[0])) * np.exp(-abs(X[1] - X[0]))),
                       ((1 + abs(X[1] - X[1])) * np.exp(-abs(X[1] - X[1]))),
                       ((1 + abs(X[1] - X[2])) * np.exp(-abs(X[2] - X[1])))]])
        Phi_l_1 = K @ A_l_1   
        
        S = np.vstack((np.exp(X[0:4]), X[0:4] * np.exp(X[0:4]), np.exp(-X[0:4]))).T
        Q, _ = np.linalg.qr(S, mode='complete')
        A_l_2 = Q[:, -1] / h
        K = np.array([[((1 + abs(X[0] - X[0])) * np.exp(-abs(X[0] - X[0]))),
                       ((1 + abs(X[0] - X[1])) * np.exp(-abs(X[0] - X[1]))),
                       ((1 + abs(X[0] - X[2])) * np.exp(-abs(X[0] - X[2]))),
                       ((1 + abs(X[0] - X[3])) * np.exp(-abs(X[0] - X[3])))],
                      [((1 + abs(X[1] - X[0])) * np.exp(-abs(X[1] - X[0]))),
                       ((1 + abs(X[1] - X[1])) * np.exp(-abs(X[1] - X[1]))),
                       ((1 + abs(X[1] - X[2])) * np.exp(-abs(X[1] - X[2]))),
                       ((1 + abs(X[1] - X[3])) * np.exp(-abs(X[1] - X[3])))],
                      [((1 + abs(X[2] - X[0])) * np.exp(-abs(X[2] - X[0]))),
                       ((1 + abs(X[2] - X[1])) * np.exp(-abs(X[2] - X[1]))), 
                       ((1 + abs(X[2] - X[2])) * np.exp(-abs(X[2] - X[2]))),
                       ((1 + abs(X[2] - X[3])) * np.exp(-abs(X[2] - X[3])))]])
        Phi_l_2 = K @ A_l_2
        
        A_central = np.zeros((5, n - 4))
        Phi_central = np.zeros((3, n - 4))
        
        for i in range(3, n - 1):
            S = np.vstack((np.exp(X[i - 3:i + 2]),
                           np.exp(-X[i - 3:i + 2]),
                           X[i - 3:i + 2] * np.exp(X[i - 3:i + 2]),
                           X[i - 3:i + 2] * np.exp(-X[i - 3:i + 2]))).T
            Q, _ = np.linalg.qr(S, mode='complete')
            A_central[:, i - 3] = -Q[:, -1] / h
            K = np.array([[((1 + abs(X[i - 2] - X[i - 3])) * np.exp(-abs(X[i - 2] - X[i - 3]))),
                           ((1 + abs(X[i - 2] - X[i - 2])) * np.exp(-abs(X[i - 2] - X[i - 2]))),
                           ((1 + abs(X[i - 2] - X[i - 1])) * np.exp(-abs(X[i - 2] - X[i - 1]))),
                           ((1 + abs(X[i - 2] - X[i])) * np.exp(-abs(X[i - 2] - X[i]))),
                           ((1 + abs(X[i - 2] - X[i + 1])) * np.exp(-abs(X[i - 2] - X[i + 1])))],
                          [((1 + abs(X[i - 1] - X[i - 3])) * np.exp(-abs(X[i - 1] - X[i - 3]))),
                           ((1 + abs(X[i - 1] - X[i - 2])) * np.exp(-abs(X[i - 1] - X[i - 2]))),
                           ((1 + abs(X[i - 1] - X[i - 1])) * np.exp(-abs(X[i - 1] - X[i - 1]))),
                           ((1 + abs(X[i - 1] - X[i])) * np.exp(-abs(X[i - 1] - X[i]))),
                           ((1 + abs(X[i - 1] - X[i + 1])) * np.exp(-abs(X[i - 1] - X[i + 1])))],
                          [((1 + abs(X[i] - X[i - 3])) * np.exp(-abs(X[i] - X[i - 3]))),
                           ((1 + abs(X[i] - X[i - 2])) * np.exp(-abs(X[i] - X[i - 2]))),
                           ((1 + abs(X[i] - X[i - 1])) * np.exp(-abs(X[i] - X[i - 1]))),
                           ((1 + abs(X[i] - X[i])) * np.exp(-abs(X[i] - X[i]))),
                           ((1 + abs(X[i] - X[i + 1])) * np.exp(-abs(X[i] - X[i + 1])))]] )
            Phi_central[:, i - 3] = K @ A_central[:, i - 3]
        
        S = np.vstack((np.exp(-X[n - 4:n]),
                       X[n - 4:n] * np.exp(-X[n - 4:n]),
                       np.exp(X[n - 4:n]))).T
        Q, _ = np.linalg.qr(S, mode='complete')
        A_r_2 = Q[:, -1] / h
        K = np.array([[((1 + abs(X[n - 3] - X[n - 4])) * np.exp(-abs(X[n - 3] - X[n - 4]))),
                       ((1 + abs(X[n - 3] - X[n - 3])) * np.exp(-abs(X[n - 3] - X[n - 3]))),
                       ((1 + abs(X[n - 3] - X[n - 2])) * np.exp(-abs(X[n - 3] - X[n - 2]))),
                       ((1 + abs(X[n - 3] - X[n - 1])) * np.exp(-abs(X[n - 3] - X[n - 1])))],
                      [((1 + abs(X[n - 2] - X[n - 4])) * np.exp(-abs(X[n - 2] - X[n - 4]))),
                       ((1 + abs(X[n - 2] - X[n - 3])) * np.exp(-abs(X[n - 2] - X[n - 3]))),
                       ((1 + abs(X[n - 2] - X[n - 2])) * np.exp(-abs(X[n - 2] - X[n - 2]))),
                       ((1 + abs(X[n - 2] - X[n - 1])) * np.exp(-abs(X[n - 2] - X[n - 1])))],
                      [((1 + abs(X[n - 1] - X[n - 4])) * np.exp(-abs(X[n - 1] - X[n - 4]))),
                       ((1 + abs(X[n - 1] - X[n - 3])) * np.exp(-abs(X[n - 1] - X[n - 3]))),
                       ((1 + abs(X[n - 1] - X[n - 2])) * np.exp(-abs(X[n - 1] - X[n - 2]))),
                       ((1 + abs(X[n - 1] - X[n - 1])) * np.exp(-abs(X[n - 1] - X[n - 1])))]] )
        Phi_r_2 = K @ A_r_2
        
        S = np.vstack((np.exp(-X[n - 3:n]),
                       X[n - 3:n] * np.exp(-X[n - 3:n]),
                       np.exp(X[n - 3:n]))).T
        Q, _ = np.linalg.qr(S, mode='complete')
        A_r_1 = Q[:, -1] / h
        K = np.array([[((1 + abs(X[n - 2] - X[n - 3])) * np.exp(-abs(X[n - 2] - X[n - 3]))),
                       ((1 + abs(X[n - 2] - X[n - 2])) * np.exp(-abs(X[n - 2] - X[n - 2]))),
                       ((1 + abs(X[n - 2] - X[n - 1])) * np.exp(-abs(X[n - 2] - X[n - 1])))],
                      [((1 + abs(X[n - 1] - X[n - 3])) * np.exp(-abs(X[n - 1] - X[n - 3]))),
                       ((1 + abs(X[n - 1] - X[n - 2])) * np.exp(-abs(X[n - 1] - X[n - 2]))),
                       ((1 + abs(X[n - 1] - X[n - 1])) * np.exp(-abs(X[n - 1] - X[n - 1])))]] )
        Phi_r_1 = K @ A_r_1
        
        A1 = A_central[2, :]
        A1 = np.insert(A1, 0, A_l_2[1])
        A1 = np.insert(A1, 0, A_l_1[0])
        A1 = np.insert(A1, len(A1), A_r_2[-2])
        A1 = np.insert(A1, len(A1), A_r_1[-1])
        
        A2 = A_central[3, :]
        A2 = np.insert(A2, 0, A_l_2[2])
        A2 = np.insert(A2, 0, A_l_1[1])
        A2 = np.insert(A2, len(A2), A_r_2[-1])
        
        A3 = A_central[4, :]
        A3 = np.insert(A3, 0, A_l_2[-1])
        A3 = np.insert(A3, 0, A_l_1[-1])
        
        A4 = A_central[1, :]
        A4 = np.insert(A4, 0, A_l_2[0])
        A4 = np.insert(A4, len(A4), A_r_2[1])
        A4 = np.insert(A4, len(A4), A_r_1[1])
        
        A5 = A_central[0, :]
        A5 = np.insert(A5, len(A5), A_r_2[0])
        A5 = np.insert(A5, len(A5), A_r_1[0])
        
        A = spdiags(A1, 0, n, n)
        A = A + spdiags(A2, -1, n, n)
        A = A + spdiags(A3, -2, n, n)
        A = A + spdiags(A4, -1, n, n).transpose()
        A = A + spdiags(A5, -2, n, n).transpose()
        
        A_csr = A.tocsr()  
        
        P1 = Phi_central[1, :]
        P1 = np.insert(P1, 0, Phi_l_2[1])
        P1 = np.insert(P1, 0, Phi_l_1[0])
        P1 = np.insert(P1, len(P1), Phi_r_2[-2])
        P1 = np.insert(P1, len(P1), Phi_r_1[-1])
        
        P2 = Phi_central[2, :]
        P2 = np.insert(P2, 0, Phi_l_2[2])
        P2 = np.insert(P2, 0, Phi_l_1[1])
        P2 = np.insert(P2, len(P2), Phi_r_2[-1])
        
        P3 = Phi_central[0, :]
        P3 = np.insert(P3, 0, Phi_l_2[0])
        P3 = np.insert(P3, len(P3), Phi_r_2[0])
        P3 = np.insert(P3, len(P3), Phi_r_1[0])
        
        Phi = spdiags(P1, 0, n, n)
        Phi = Phi + spdiags(P2, -1, n, n)
        Phi = Phi + spdiags(P3, -1, n, n).transpose()
        
        Phi_csr = Phi.tocsr()
        
        return A_csr, Phi_csr
    
    elif nu == 5/2:
        n = X.shape[0]
        h = n ** -3
        X = theta * X
        
        S = np.vstack(([np.exp(X[0:4]), X[0:4] * np.exp(X[0:4]), (X[0:4]**2) * np.exp(X[0:4])])).T
        Q, _ = np.linalg.qr(S, mode='complete')  
        A_l_1 = Q[:, -1] / h
        K = np.zeros((3, 4))
        for i in range(3):
            for j in range(4):
                K[i, j] = (1 + abs(X[i] - X[j]) + abs(X[i] - X[j])**2 / 3) * np.exp(-abs(X[i] - X[j]))
        Phi_l_1 = K @ A_l_1
        
        S = np.vstack((np.exp(X[0:5]), X[0:5]*np.exp(X[0:5]), X[0:5]**2*np.exp(X[0:5]), np.exp(-X[0:5]))).T
        Q, _ = np.linalg.qr(S, mode='complete')
        A_l_2 = Q[:, -1] / h
        K = np.zeros((4, 5))
        for i in range(4):
            for j in range(5):
                K[i, j] = (1 + abs(X[i] - X[j]) + abs(X[i] - X[j])**2 / 3) * np.exp(-abs(X[i] - X[j]))
        Phi_l_2 = K @ A_l_2
        
        S = np.vstack((np.exp(X[0:6]), X[0:6]*np.exp(X[0:6]), X[0:6]**2*np.exp(X[0:6]), np.exp(-X[0:6]), X[0:6]*np.exp(-X[0:6]))).T
        Q, _ = np.linalg.qr(S, mode='complete')
        A_l_3 = Q[:, -1] / h
        K = np.zeros((5, 6))
        for i in range(5):
            for j in range(6):
                K[i, j] = (1 + abs(X[i] - X[j]) + abs(X[i] - X[j])**2 / 3) * np.exp(-abs(X[i] - X[j]))
        Phi_l_3 = K @ A_l_3
        
        A_central = np.zeros((7, n-6)) 
        Phi_central = np.zeros((5, n-6))
        for i in range(4, n - 2):
            S = np.vstack((np.exp(X[i-4:i+3]), 
                           np.exp(-X[i-4:i+3]), 
                           X[i-4:i+3] * np.exp(X[i-4:i+3]), 
                           X[i-4:i+3] * np.exp(-X[i-4:i+3]), 
                           X[i-4:i+3]**2 * np.exp(-X[i-4:i+3]), 
                           X[i-4:i+3]**2 * np.exp(X[i-4:i+3]))).T
            Q, _ = np.linalg.qr(S, mode='complete')
            A_central[:, i-4] = -Q[:, -1] / h    
            K = np.zeros((5, 7))
            for k1 in range(i - 3, i + 2):
                for k2 in range(i - 4, i + 3):
                    K[k1 - (i - 3), k2 - (i - 4)] = (1 + abs(X[k1] - X[k2]) + abs(X[k1] - X[k2])**2 / 3) * np.exp(-abs(X[k1] - X[k2]))
            Phi_central[:,i-4] = K @ A_central[:,i-4]
        
        S = np.vstack(([np.exp(-X[n - 6: n]), 
                       X[n - 6: n] * np.exp(-X[n - 6: n]), 
                       X[n - 6: n]**2 * np.exp(-X[n - 6: n]), 
                       np.exp(X[n - 6: n]), 
                       X[n - 6: n] * np.exp(X[n - 6: n])])).T 
        Q, _ = np.linalg.qr(S, mode='complete')
        A_r_3 = Q[:, -1] / h
        K = np.zeros((5, 6))
        for i in range(5):
            for j in range(6):
                K[i, j] = (1 + abs(X[i + (n-5)] - X[j + (n-6)]) + abs(X[i + (n-5)] - X[j + (n-6)])**2 / 3) * np.exp(-abs(X[i + (n-5)] - X[j + (n-6)]))
        Phi_r_3 = K @ A_r_3
        
        S = np.vstack(([np.exp(-X[n - 5: n]), 
                       X[n - 5: n] * np.exp(-X[n - 5: n]), 
                       X[n - 5: n]**2 * np.exp(-X[n - 5: n]), 
                       np.exp(X[n - 5: n])])).T 
        Q, _ = np.linalg.qr(S, mode='complete')
        A_r_2 = Q[:, -1] / h
        K = np.zeros((4, 5))
        for i in range(4):
            for j in range(5):
                K[i, j] = (1 + abs(X[i + (n-4)] - X[j + (n-5)]) + abs(X[i + (n-4)] - X[j + (n-5)])**2 / 3) * np.exp(-abs(X[i + (n-4)] - X[j + (n-5)]))
        Phi_r_2 = K @ A_r_2
        
        S = np.vstack(([np.exp(-X[n - 4: n]), 
                       X[n - 4: n] * np.exp(-X[n - 4: n]), 
                       X[n - 4: n]**2 * np.exp(-X[n - 4: n])])).T 
        Q, _ = np.linalg.qr(S, mode='complete')  
        A_r_1 = Q[:, -1] / h
        K = np.zeros((3, 4))
        for i in range(3):
            for j in range(4):
                K[i, j] = (1 + abs(X[i + (n-3)] - X[j + (n-4)]) + abs(X[i + (n-3)] - X[j + (n-4)])**2 / 3) * np.exp(-abs(X[i + (n-3)] - X[j + (n-4)]))
        Phi_r_1 = K @ A_r_1
        
        
        A1 = A_central[3, :]
        A1 = np.insert(A1, 0, A_l_3[2])
        A1 = np.insert(A1, 0, A_l_2[1])
        A1 = np.insert(A1, 0, A_l_1[0])
        A1 = np.insert(A1, len(A1), A_r_3[-3])
        A1 = np.insert(A1, len(A1), A_r_2[-2])
        A1 = np.insert(A1, len(A1), A_r_1[-1])
        
        A2 = A_central[4, :]
        A2 = np.insert(A2, 0, A_l_3[3])
        A2 = np.insert(A2, 0, A_l_2[2])
        A2 = np.insert(A2, 0, A_l_1[1])
        A2 = np.insert(A2, len(A2), A_r_3[-2])
        A2 = np.insert(A2, len(A2), A_r_2[-1])
        
        A3 = A_central[5, :]
        A3 = np.insert(A3, 0, A_l_3[4])
        A3 = np.insert(A3, 0, A_l_2[3])
        A3 = np.insert(A3, 0, A_l_1[2])
        A3 = np.insert(A3, len(A3), A_r_3[-1])
        
        A4 = A_central[6, :]
        A4 = np.insert(A4, 0, A_l_3[5])
        A4 = np.insert(A4, 0, A_l_2[4])
        A4 = np.insert(A4, 0, A_l_1[3])
        
        A5 = A_central[2, :]
        A5 = np.insert(A5, 0, A_l_3[1])
        A5 = np.insert(A5, 0, A_l_2[0])
        A5 = np.insert(A5, len(A5), A_r_3[2])
        A5 = np.insert(A5, len(A5), A_r_2[2])
        A5 = np.insert(A5, len(A5), A_r_1[2])
        
        A6 = A_central[1, :]
        A6 = np.insert(A6, 0, A_l_3[0])
        A6 = np.insert(A6, len(A6), A_r_3[1])
        A6 = np.insert(A6, len(A6), A_r_2[1])
        A6 = np.insert(A6, len(A6), A_r_1[1])
        
        A7 = A_central[0, :]
        A7 = np.insert(A7, len(A7), A_r_3[0])
        A7 = np.insert(A7, len(A7), A_r_2[0])
        A7 = np.insert(A7, len(A7), A_r_1[0])
        
        A = spdiags(A1, 0, n, n)
        A = A + spdiags(A2, -1, n, n)
        A = A + spdiags(A3, -2, n, n)
        A = A + spdiags(A4, -3, n, n)
        A = A + spdiags(A5, -1, n, n).transpose()
        A = A + spdiags(A6, -2, n, n).transpose()
        A = A + spdiags(A7, -3, n, n).transpose()
        
        A_csr = A.tocsr()  
        
        P1 = Phi_central[2, :]
        P1 = np.insert(P1, 0, Phi_l_3[2])
        P1 = np.insert(P1, 0, Phi_l_2[1])
        P1 = np.insert(P1, 0, Phi_l_1[0])
        P1 = np.insert(P1, len(P1), Phi_r_3[-3])
        P1 = np.insert(P1, len(P1), Phi_r_2[-2])
        P1 = np.insert(P1, len(P1), Phi_r_1[-1])
        
        P2 = Phi_central[3, :]
        P2 = np.insert(P2, 0, Phi_l_3[3])
        P2 = np.insert(P2, 0, Phi_l_2[2])
        P2 = np.insert(P2, 0, Phi_l_1[1])
        P2 = np.insert(P2, len(P2), Phi_r_3[-2])
        P2 = np.insert(P2, len(P2), Phi_r_2[-1])
        
        P3 = Phi_central[4, :]
        P3 = np.insert(P3, 0, Phi_l_3[4])
        P3 = np.insert(P3, 0, Phi_l_2[3])
        P3 = np.insert(P3, 0, Phi_l_1[2])
        P3 = np.insert(P3, len(P3), Phi_r_3[-1])
        
        P4 = Phi_central[1, :]
        P4 = np.insert(P4, 0, Phi_l_3[1])
        P4 = np.insert(P4, 0, Phi_l_2[0])
        P4 = np.insert(P4, len(P4), Phi_r_3[1])
        P4 = np.insert(P4, len(P4), Phi_r_2[1])
        P4 = np.insert(P4, len(P4), Phi_r_1[1])
        
        P5 = Phi_central[0, :]
        P5 = np.insert(P5, 0, Phi_l_3[0])
        P5 = np.insert(P5, len(P5), Phi_r_3[0])
        P5 = np.insert(P5, len(P5), Phi_r_2[0])
        P5 = np.insert(P5, len(P5), Phi_r_1[0])
        
        Phi = spdiags(P1, 0, n, n)
        Phi = Phi + spdiags(P2, -1, n, n)
        Phi = Phi + spdiags(P3, -2, n, n)
        Phi = Phi + spdiags(P4, -1, n, n).transpose()
        Phi = Phi + spdiags(P5, -2, n, n).transpose()

        Phi_csr = Phi.tocsr()
        
        return A_csr, Phi_csr

