import numpy as np
from scipy.linalg import svd, lstsq, norm
from sklearn import linear_model
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from typing import List
import cvxpy as cp

def lsq(X_tilde, Y):

    W = []

    _, n_species = Y.shape 

    for i in range(n_species):

        w, _, _, _ = lstsq(X_tilde, Y[:,i])
        relres = norm(Y[:,i] - X_tilde@w)/norm(Y[:,i])
        W.append(w)
        print("\nequation #"+str(i))
        print('relative residual', relres)
        print('coefficients', w)

    return np.array(W)


def lasso(X_tilde, Y, alpha_lasso):

    W = []
    _, n_species = Y.shape 


    for i in range(n_species):

        clf = linear_model.Lasso(alpha= alpha_lasso)
        clf.fit(X_tilde, Y[:,i])
        w = clf.coef_
        print(clf)
        #x, _, _, _ = lstsq(D, YD[:,i])
        relres = norm(Y[:,i] - X_tilde@w)/norm(Y[:,i])

        print("\nequation #"+str(i))
        print('relative residual', relres)
        print('coefficients', w)
        W.append(w) 

    return np.array(W)


#CVXP implementation of Lasso 

def loss_fn(X, Y, beta):
    return cp.norm2(X @ beta - Y)**2

def regularizer(beta):
    return cp.norm1(beta)

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)

def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value


def lasso_cvxp(X_tilde, Y, alpha_lasso):

    W = []
    _, n_species = Y.shape 


    for i in range(n_species):
        W_i = cp.Variable((14,))
        lambd = cp.Parameter(nonneg=True)
        problem = cp.Problem(cp.Minimize(objective_fn(X_tilde, Y[:,i], W_i, lambd)))
        
        lambd.value = alpha_lasso
        problem.solve()
        #train_errors.append(mse(X_tilde, Y[:,i], W_i))
        #test_errors.append(mse(X_tilde, Y_gt[:,i], W_i))
        W.append(W_i.value)

    return np.array(W)






def stlsq(X_tilde, Y, alpha): 
    _, n_species = Y.shape
    _, m_features = X_tilde.shape
    W = np.zeros([n_species, m_features])
    #print(W.shape)
    for i in range (n_species):  
        idx = np.arange(0, X_tilde.shape[1]) # indices left
        ind = np.array([0,0]) # removed indices 
        while np.size(ind) > 0:
            w, res, rnk, s = lstsq(X_tilde[:,idx], Y[:, i])
            ind = np.where(np.abs(w)< alpha)
            #print("ind:", ind)
            idx = np.delete(idx, ind)
            #print(w)
            w = np.array(w).T 
            # print(idx)
            # print(w.shape)
            ind = np.array(ind)
        #print(W[i,idx])
        W[i,idx] = w   
        #W.append(w .T) 
        #print("equation ", i+1)
    return W


# def stlsq_find_small(W, X_tilde, alpha, Y, trig):
#     k_time, m_features = X_tilde.shape
#     c = 0
#     for j in range(W.size):
#         if(abs(W[j])<=alpha and W[j]!=0):
#             #remove corresponding column from X_tilde 
#             W[j] = 0
#             X_tilde_new = np.delete(X_tilde, j, 1)
#             new_column = [0]
#             X_tilde_new = np.insert(X_tilde_new, j, new_column, axis=1)
#         else: 
#             c+=1 
#     #print(c)
#     if (c == W.size):
#         trig = False 
#         print('converged')
#         return X_tilde, W, trig
#     else: return X_tilde_new, W, trig


# def stlsq(X_tilde, Y, alpha):
#     k_times, n_species = Y.shape
#     _, m_features = X_tilde.shape
#     W = np.zeros((n_species, m_features))
#     for i in range(n_species): 
#         # solve for each equation separately: 
#         w, _, _, _ = lstsq(X_tilde, Y[:,i])
#         trig = True
#         X_tilde_new, w, trig = stlsq_find_small(w, X_tilde, alpha, Y[:,i], trig)
#         while(trig == True):
#             w, _, _, _ = lstsq(X_tilde_new, Y[:,i])
#             X_tilde_new, w, trig = stlsq_find_small(w, X_tilde, alpha, Y[:,i], trig)
#         W[i] = w

#     return W     



def ridge(X_tilde, Y, alpha_ridge):

    W = []
    _, n_species = Y.shape 


    for i in range(n_species):

        clf = linear_model.Ridge(alpha= alpha_ridge)
        clf.fit(X_tilde, Y[:,i])
        w = clf.coef_
        print(clf)
        #x, _, _, _ = lstsq(D, YD[:,i])
        relres = norm(Y[:,i] - X_tilde@w)/norm(Y[:,i])

        print("\nequation #"+str(i))
        print('relative residual', relres)
        print('coefficients', w)
        W.append(w) 

    return np.array(W)

    
def plot_regularization_path(alphas: np.ndarray, W: List[np.ndarray], W0: np.ndarray, i: int, labels: List[str]):
    """
    Args:
        alphas (np.ndarray): list of generated optimization parameter 
        W (List[np.ndarray]): list of weights corresponding to every alpha.
            Every weiths matrix have dimension  number of equations x len(features) 
        W0 (np.ndarray): ground truth weights (dimension  number of equations x len(features))
        i (int): requested index of equation for visualization
        labels (List[str]): names of features
    """
    n_equations, n_features = W[0].shape
    # colors = sns.color_palette('husl', n_colors=ncolors)  # a list of RGB tuples

    zero_ind = np.where(np.abs(W0[i]) == 0) 
    nonzero_ind = np.where(np.abs(W0[i]) != 0) 
    print(nonzero_ind)
    print(zero_ind)

    labels = np.array(labels)
    
    legend_labels = np.concatenate((labels[nonzero_ind], labels[zero_ind]), axis=None)

    # for equation_id in nonzero_ind:
    #     non_zero_weights = [wi[i, equation_id] for wi in W]
    #     plt.plot(alphas, non_zero_weights , '-', linewidth=4)
    # for equation_id in zero_ind:
    #     zero_weights = [wi[i, equation_id] for wi in W]
    #     plt.plot(alphas, zero_weights, '--', linewidth=3)
    colormap = plt.cm.rainbow
    cycler = plt.cycler('color', colormap(np.linspace(0, 1, n_features)))
    plt.gca().set_prop_cycle(cycler)

    plt.semilogx(alphas, [wi[i][nonzero_ind] for wi in W], '-', linewidth=4)
    plt.semilogx(alphas, [wi[i][zero_ind] for wi in W], '--', linewidth=3)

    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$w_{i %i}$" %i)
    #plt.xscale("log")
    plt.title("Regularization Path for %i equation" %i)
    plt.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend(legend_labels,loc="upper right")


        