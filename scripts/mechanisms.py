#This script contains functions for generating data of different kinetic mechanisms


import pysindy as ps
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.interpolate import UnivariateSpline
from scipy.linalg import norm



def ode_A1r(t, U,  constants):
    A, P, cat, catA = U
    k1, k_1, k2, k_2 = constants

    dAdt = - k1 * A * cat + k_1 * catA
    dPdt = k2 * catA - k_2 * cat * P
    dcatdt = - k1 * A * cat + k2 * catA + k_1 * catA - k_2 * cat * P
    dcatAdt = k1 * A * cat - k2 * catA - k_1 * catA + k_2 * cat * P

    return np.array([dAdt, dPdt, dcatdt, dcatAdt])

def ode_A1ra(t, U,  constants):
    A, P, cat, catact, catactA = U
    k1, k_1, k2, k_2, ka = constants

    dAdt = - k1 * A * catact + k_1 * catactA
    dPdt = k2 * catactA - k_2 * catact * P
    dcatdt = -ka * cat
    dcatactdt = - k1 * A * catact + k2 * catactA + k_1 * catactA - k_2 * catact * P + ka * cat
    dcatactAdt = k1 * A * catact - k2 * catactA - k_1 * catactA + k_2 * catact * P

    return np.array([dAdt, dPdt, dcatdt, dcatactdt, dcatactAdt])


def ode_A2r(t, U,  constants):
    A, P, cat, catA, B = U
    k1, k_1, k2, k_2 = constants
       
    dAdt = -k1*A*cat + k_1*catA*B
    dPdt = k2*catA*B - k_2*cat*P
    dcatdt = -k1*A*cat + k_1*catA*B + k2*catA*B - k_2*cat*P
    dcatAdt = k1*A*cat - k_1*catA*B - k2*catA*B + k_2*cat*P
    dBdt = k1*A*cat - k_1*catA*B - k2*catA*B + k_2*cat*P

    return np.array([dAdt, dPdt, dcatdt, dcatAdt, dBdt])



def ode(t, U, constants, mechtype):
    """
    Args:
        t: time-points vector
        U: initial conditions for concentrations
        constants: vector of constants k1, k2 ... 
        mechtype(string): string for type of mechanism

    Returns:
        np.array: system of ODEs
    """
 
    if mechtype =='A1r': 
        return ode_A1r(t, U, constants)

    elif mechtype =='A2r': 
        return ode_A2r(t, U, constants)

    elif mechtype =='A1ra': 
        return ode_A1ra(t, U, constants)

    else: 
        raise ValueError("Can not recognise input type of mechanism")




#########################################
def get_species_number(mechtype):
    if mechtype == 'A1r': 
        n_species = 4
    elif mechtype == 'A2r' or 'A1ra':
        n_species = 5
    else: 
        raise ValueError("Can not recognise input type of mechanism")

    return n_species


################### Derivatives approximation ######################
def dspline(t_eval, y):
    yd = y.copy()
    for i in range(y.shape[0]):
        ys = UnivariateSpline(t_eval, y[i,:], k=3, s=0) # cubic spline
        yd[i,:] = ys.derivative()(t_eval)
    return yd


def fdm(t, Y):

    k, n = Y.shape
    yd = np.zeros((0,n))
    
    print(yd.shape)

    for y in Y:
        res = []


        
        c2 = ((y[1] - y[0])*t[2] + (y[0]-y[2])*t[1])/(t[1]*t[1]*t[2] - t[1]*t[2]*t[2]) 
        c1 = y[1]/t[1] - y[0]/t[1] - c2*t[1]
        res.append(c1)

        # h1 = t[1] - t[0] 
        # h2 = t[2] - t[0]
        # c1 = -(h1 + h2)/(h1*h2)* y[0] + h2/(h1*h2 - h1**2)*y[1] - h1/(h2**2 - h1*h2) *y[2]
        # #print("p0", c1)
        # res.append(c1)
        
        n = t.size-1
        for i in range(1, n):
            h1 = t[i] - t[i-1] 
            h2 = t[i+1] - t[i-1]
            
            c1 = -(h1 + h2)/(h1*h2)* y[i-1] + h2/(h1*h2 - h1**2)*y[i] - h1/(h2**2 - h1*h2) *y[i+1]
            c2 = 1/(h1*h2) * y[i-1] + 1/(h1**2 - h1*h2)*y[i] - 1/(h1*h2 - h2**2)*y[i+1]
            #pdt = 2*c2 + c1
            pdt = c1
            res.append(pdt)
        #backward difference 
        last = (y[n] - y[n-1])/(t[n] - t[n-1])
        res.append(last)
        #print("res length:", len(res))
        #yd[i, :] = np.array(res)

    
        yd = np.vstack((yd, np.array(res)))
    return yd 



def fdm_c1(t,Y):
    k, n = Y.shape
    yd = np.zeros((0,n))
    for y in Y:
        res = []
        
        h1 = t[1] - t[0] 
        h2 = t[2] - t[0]
        c1 = -(h1 + h2)/(h1*h2)* y[0] + h2/(h1*h2 - h1**2)*y[1] - h1/(h2**2 - h1*h2) *y[2]
        res.append(c1)
        
        n = t.size-1
        for i in range(1, n):
            h1 = t[i] - t[i-1] 
            h2 = t[i+1] - t[i-1]
            
            c1 = -(h1 + h2)/(h1*h2)* y[i-1] + h2/(h1*h2 - h1**2)*y[i] - h1/(h2**2 - h1*h2) *y[i+1]
            pdt = c1
            res.append(pdt)
        
        #backward difference 
        last = (y[n] - y[n-1])/(t[n] - t[n-1])
        
        res.append(last)   
        yd = np.vstack((yd, np.array(res)))
    return yd     




################### solution vectors generation  ######################
def find_solutions(U, constants, mechtype, approxtype, tmax,  rtol, integration, t_eval = None):
    """
    Args:
        U(np.ndarray): initial conditions for concentrations
        constants (List(float)): vector of constants k1, k2 ... 
        mechtype(string): string for type of mechanism
        approxtype(string): string for type of derivatives approximation (spline or fdm)
        tmax(int): last time value
        npts(int): number of points in time-points vector
        rtol: noise tolerance 
        integration (str): method for integration 

    Returns:
        np.array: X - solutions matrix
        np.array: T - time-points vector
        np.array: YD - approximation of derivatives 
    """


    #t_eval = np.linspace(0,tmax,npts) 

    # if non_equispaced is True: 
    #     t_eval = (1-np.cos(np.linspace(0,np.pi,npts)))/2 * tmax # Chebyshev pts


    n_species = get_species_number(mechtype)


    T = np.zeros((0,))   # time points
    X = np.zeros((0,n_species)) 
    YD = np.zeros((0,n_species)) # derivatives

    


    for u0 in U:

        sol = integrate.solve_ivp(fun=lambda t,U: ode(t,U,constants, mechtype), t_span=(0, tmax), y0=u0, method = integration, t_eval=t_eval, rtol=rtol)
        #sol = integrate.solve_ivp(fun=lambda t,U: ode(t,U,constants, mechtype), t_span=t_eval[[0,-1]], y0=u0, t_eval=t_eval, rtol=1e-10)
        print(integration)
        if(approxtype == 'spline'):
            sol.yd = dspline(sol.t, sol.y)
        elif(approxtype == 'fdm'):
            #sol.yd = fdm(sol.t, sol.y)
            sol.yd = fdm_c1(sol.t, sol.y)

        X = np.vstack((X, sol.y.T))

        T = np.hstack((T, sol.t))
        YD = np.vstack((YD, sol.yd.T))

    return X, T, YD 

















def get_labels(mechtype):
    if mechtype == 'A1r': 
        return ['A','P','cat','catA']
    elif mechtype == 'A2r': 
        return ['A', 'P', 'cat', 'catA', 'B']
    elif mechtype == 'A1ra':
        return ['A', 'P', 'cat', 'cat_act', 'catA_act']
    else: 
        raise ValueError("Can not recognise input type of mechanism")



################# Generate features #######################
#without square terms 
# def get_features(X, mechtype):
#     labels = get_labels(mechtype)
#     labels_tilde = labels.copy()
#     X_tilde = X.copy()

#     n_species = get_species_number(mechtype)

#     for i in range(n_species): # append pairwise products, omitting squared terms and opposite order terms
#         for j in range(i): 
#             xi = X[:,i].reshape(1,-1).T # silly but needed because Y[:,i] is 1-dimensional (while Y[i,:] is 2-dimensional)
#             xj = X[:,j].reshape(1,-1).T

#             X_tilde = np.hstack((X_tilde, xi*xj)) 
#             labels_tilde.append(labels[i]+"*"+labels[j])
        
#     print('X_tilde features matrix shape: ', X_tilde.shape)
#     print('labels: ', labels_tilde)

#     return X_tilde, labels_tilde 



def get_features(X: np.ndarray, mechtype: str, square=False):
    '''
    Args:
        X (np.ndarray): input data
        mechtype (str): type of mechanism
        square (bool): If true add squared term. Defaults is False
    '''
    labels = get_labels(mechtype)
    labels_tilde = labels.copy()
    X_tilde = X.copy()

    n_species = get_species_number(mechtype)

    for i in range(n_species): # append pairwise products, omitting squared terms and opposite order terms
        for j in range(i): 
            xi = X[:,i].reshape(1,-1).T # silly but needed because Y[:,i] is 1-dimensional (while Y[i,:] is 2-dimensional)
            xj = X[:,j].reshape(1,-1).T

            X_tilde = np.hstack((X_tilde, xi*xj)) 
            labels_tilde.append(labels[i]+"*"+labels[j])

    if square: 
        print('added squared terms')
        for i in range(n_species):
            xi = X[:,i].reshape(1,-1).T 
            X_tilde = np.hstack((X_tilde, xi*xi)) 
            labels_tilde.append(labels[i]+"*"+ labels[i])


    print('X_tilde features matrix shape: ', X_tilde.shape)
    print('labels: ', labels_tilde)

    return X_tilde, labels_tilde 




############### Plot graphs ##############################

def plot_concentrations(T, X, mechtype):
    plt.plot(T, X, '-o')
    labels = get_labels(mechtype)
    plt.legend(labels)
    plt.title('Species Concentrations vs. Time')
    plt.xlabel('time $t$')
    plt.ylabel('concentration')




def plot_derivatives(T, Y, mechtype):
    plt.plot(T, Y, '-o')
    labels = get_labels(mechtype)
    dt_labels =[]
    for label in labels:
        dt_labels.append('$\dot{' + label + '}$')

    plt.legend(dt_labels)
    plt.title('Derivatives of Species Concentrations')
    plt.xlabel('time $t$')




############### Construct string equations from data matrices ##############################

def string_equation(labels, W, tol):
    equations = []
    for coefs in W:
        equation = ''
        for coef, label in zip(coefs, labels):
            if abs(coef)<tol: 
                continue
            eq = ''
            if coef < 0:

                eq += str(coef) + '*' +  str(label)
            elif coef >=0:
                eq += '+' + str(coef) + '*' +  str(label)
            equation += eq 
        print(equation)
        equations.append(equation)
        print('______')
    return equations 

def print_equation(labels, W, tol):
    equations = []
    s = 0
    for coefs in W:
        equation = ''
        for coef, label in zip(coefs, labels): 

            if abs(coef)<tol: 
                continue
            eq = ''
            if coef < 0:
                s+=1
                eq += ("%.2f" % coef) + '[' +  str(label) + ']'
            elif coef >=0:
                s+=1
                eq += '+' + ("%.2f" % coef) + '[' +  str(label) + ']'
            equation += eq 
        print(equation)
        equations.append(equation)
        print('______')
    print('sparsity:', s)
    return s 

        
def ode_from_string(t, U, equations, mechtype):
    odes = []

    if mechtype == 'A1r':
        A, P, cat, catA = U
    elif mechtype == 'A2r': 
        A, P, cat, catA, B = U
    elif mechtype == 'A1ra':
        A, P, cat, cat_act, catA_act = U 
    else:
        raise ValueError("Can not recognise input type of mechanism")

    for eq in equations: 
        if not eq: 
            odes.append(0)
        else:     
            odes.append(eval(eq))
    
    return np.array(odes)






################### solution vectors generation  ######################
def data_from_equations(U0, constants, mechtype, equations, approxtype, tmax, rtol, integration, time_points):
    
    #t_eval = np.linspace(0,tmax,npts)

    # if non_equispaced is True: 
    #     t_eval = (1-np.cos(np.linspace(0,np.pi,npts)))/2 * tmax # Chebyshev pts
        
    n_species = get_species_number(mechtype)


    T = np.zeros((0,))   # time points
    X = np.zeros((0,n_species)) 
    YD = np.zeros((0,n_species)) # derivatives
    Ygt = np.zeros((0,n_species))

    for u0 in U0:
        sol = integrate.solve_ivp(fun=lambda t,U: ode_from_string(t, U, equations, mechtype), t_span = (0, tmax), t_eval = time_points, y0=u0, method = integration, rtol=rtol)
    
        if(approxtype == 'spline'):
            yd = dspline(sol.t, sol.y)
            #print('yd: ', yd)

        elif(approxtype == 'fdm'): 
            #yd = fdm(sol.t, sol.y)
            yd = fdm_c1(sol.t, sol.y)
            #print('yd: ', yd)

        X = np.vstack((X, sol.y.T))
        T = np.hstack((T, sol.t))
        YD = np.vstack((YD, yd.T))

    Ygt =  ode(T, X.T, constants, 'A1r').T

    return X, T, Ygt 



def get_residual(YD, Y_gt):
    absres = norm(YD - Y_gt)
    print('absolute residual', absres)
    relres = norm(YD -  Y_gt)/norm(YD)
    print('relative residual', relres)
    return absres, relres 

