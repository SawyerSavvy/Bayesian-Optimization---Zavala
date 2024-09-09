import myBO
import BO_algos
import numpy as np
from math import sqrt
from joblib import Parallel, delayed
from scipy.optimize import minimize, Bounds, approx_fprime, NonlinearConstraint
import sklearn.gaussian_process as gpr
import matplotlib.pyplot as pyp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import itertools
import pandas as pd
import time

#importing myAlgae1
import os
file_path = os.path.abspath('/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Algae Model')
import sys
sys.path.append(file_path)
import Photo_mechanistic_modeling_biomass as PM
import Photo_modified as PM_N
import Photo_high_dim as PM_5
import Photo_4_dim as PM_4
import Photo_3_dim as PM_3
import Photo_2_dim as PM_2
from Grid_search_for_opt import iterate, create_axis, test_algae_BO_2D_Nitrogen

def Photo():
    '''
    Runs Bayesian Optimization for the algae model only dependent upon light intensity
    '''
    #Bounds
    bnds = Bounds((0),(1))
    #Upper limit
    ub = np.array([1000])
    #Lower limit
    lb = np.array([0])
    #Step intervals
    step = np.array([5])
    #Dimesnions
    dim = 1

    num = 100
    xs = np.empty((num,dim))
    ys = np.empty((num,1))
    ymax = np.empty((num,1))
    iteration = np.empty((num,1),dtype = int)
    xmax = np.empty((num,dim))
    intialx = np.empty((num,dim))

    grid = iterate(create_axis(dim,ub,lb,step),dim)

    kernel = gpr.kernels.Matern((1),(1,10), nu = 2.5)
    
    iterations = 10

    exp_w = 2.576

    for i in range(num):

        BO_obj = myBO.BO(ub,lb,step,dim = dim,exp_w = exp_w, kernel = kernel,system = retrieve_data,bounds = bnds)  
        BO_obj.optimizergp_man(iterations,1,grid = grid) 
        ys[i] = BO_obj.ygp[-1]
        xs[i] = BO_obj.xgp[-1]
        ymax[i] = np.max(BO_obj.ygp)
        iteration[i] = int(np.argmax(BO_obj.ygp))
        xmax[i] = BO_obj.xgp[iteration[i]]

    un = np.unique(xs)
    print(max(ys))
    pyp.hist(xs,color = 'skyblue', bins = un)
    headers = ['Last Index Time-Integrated Biomass (g-hr/L)','Last Index Light Intensity (umol photons $m^-$$^2$ $s^-$$^1$)', 'First max Time-Integrated Biomass (g-hr/L)', 'First Max Light Intensity (umol photons $m^-$$^2$ $s^-$$^1$)','First max index']
    df = pd.DataFrame(np.hstack([ys,xs,ymax,xmax,iteration]), columns = headers)
    csv = "/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/1-D/BO_iterations_Noise_" + str(iterations + 1) + "_Repeated_" + str(num) + "_exploration_" + str(exp_w) + ".csv"
    df.to_csv(csv)
    pyp.ylabel('Frequency')
    pyp.xlabel('Light Intensity (umol photons $m^-$$^2$ $s^-$$^1$)')
    #pyp.show()

def retrieve_data(light):
    light = light[0]
    print(light)
    
    df = pd.read_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/BO Algos/Data/Noise_1000_light_gaussian.csv")
    print(int(light)/5)
    return df.loc[int(light)/5,'Biomass']

def Photo_N():
    '''
    Runs Bayesian Optimization for the algae model dependent upon light intensity & nitrogen
    '''
    #Bounds
    bnds = Bounds((0,0),(1,1))
    #Upper limit
    ub = np.array([2000,400])
    #Lower limit
    lb = np.array([0,0])
    #Step intervals
    step = np.array([20,10])
    #Dimesnions
    dim = 2

    num = 100
    xs = np.empty((num,dim))
    xs_s = np.empty((num,dim))
    ys = np.empty((num,1))
    ys_s = np.empty((num,1))
    ymax = np.empty((num,1))
    iteration = np.empty((num,1),dtype = int)
    xmax = np.empty((num,dim))
    intialx = np.empty((num,dim))

    kernel = gpr.kernels.Matern((1),(1,10), nu = 2.5)
    
    iterations = 14

    exp_w = 2.576

    test_algae_BO_2D_Nitrogen()

    for i in range(num):

        BO_obj = myBO.BO(ub,lb,step,dim = dim,exp_w = exp_w,kernel = kernel,system = PM_N.BO_intializer,bounds = bnds)  
        BO_obj.optimizergp_man(iterations,1,None) 
        ys[i] = BO_obj.ygp[-1]
        xs[i] = BO_obj.xgp[-1]
        ymax[i] = np.max(BO_obj.ygp)
        iteration[i] = int(np.argmax(BO_obj.ygp))
        xmax[i] = BO_obj.xgp[iteration[i]]
        xs_s[i] = BO_obj.xgp[0]
        ys_s[i] = BO_obj.ygp[0]
    
    axis_x = np.arange(0,1000,5)
    axis_y = np.linspace(0,200,10)
    pyp.title('100 runs of nonparallel BO each with 14 iterations and 1 starting sample, no noise')
    X, Y = np.meshgrid(axis_x, axis_y)
    pyp.scatter(xmax[:,0],xmax[:,1],color = 'blue',zorder=5,label = 'Maximum')
    pyp.scatter(xs_s[:,0],xs_s[:,1],color = 'purple',zorder=4, label = 'Starting')
    pyp.xlabel('Light Intensity')
    pyp.ylabel('Nitrogen Concentration')
    pyp.show()
    
    '''
    un = np.unique(xs)
    print(max(ys))
    pyp.hist(xs,color = 'skyblue', bins = un)
    '''
    headers = ['Last Index Time-Integrated Biomass (g-hr/L)','Last Index Light Intensity (umol photons $m^-$$^2$ $s^-$$^1$)','Last Index Nitrogen','First max Time-Integrated Biomass (g-hr/L)', 'First Max Light Intensity (umol photons $m^-$$^2$ $s^-$$^1$)','First Max Nitrogen','First max index','starting point light','starting point nitrogen','ending point']
    df = pd.DataFrame(np.hstack([ys,xs,ymax,xmax,iteration,xs_s,ys_s]), columns = headers)
    csv = "/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/2-D/BO/BO_Noise_iterations_" + str(iterations + 1) + "_Repeated_" + str(num) + "_exploration_" + str(exp_w) + ".csv"
    df.to_csv(csv)

def Photo_N_arg_test():
    '''
    Test arguments for the algae model dependent upon light intensity & nitrogen
    '''
     
    test_array = np.array([1381,201])

    soly,solt = PM_N.it(test_array)
    
    fig, axs = pyp.subplots(3)

    axs[0].scatter(solt,soly[0,:])
    axs[0].set_ylabel('Biomass')
    axs[1].scatter(solt,soly[1,:])
    axs[1].set_ylabel('N Concentration')
    axs[2].scatter(solt,soly[2,:])
    axs[2].set_ylabel('Quota')
    axs[2].set_xlabel('Time')
    pyp.show()

def high_dim_basic_BO():
    '''
    Runs Bayesian Optimization for the algae model dependent upon light intensity, nitrogen, nitrogen qouta, light path dimension, sun light hours, and batch time
    '''
    #Bounds
    bnds = Bounds((0,0,0,0,0),(1,1,1,1,1))
    #Upper limit
    ub = np.array([2000,400,80,100,20])
    #Lower limit
    lb = np.array([1,1,20,10,12])
    #Step intervals
    step = np.array([20,10,6,5,1])
    #Dimensions
    dim = 5

    num = 30
    xs = np.empty((num,dim))
    xs_s = np.empty((num,dim))
    ys = np.empty((num,1))
    ys_s = np.empty((num,1))
    ymax = np.empty((num,1))
    iteration = np.empty((num,1),dtype = int)
    xmax = np.empty((num,dim))
    intialx = np.empty((num,dim))

    kernel = gpr.kernels.Matern((1),(1,10), nu = 2.5)
    
    iterations = 200

    exp_w = 2.576
    picks_per_it = 1
    initial_x = 10

    results = np.empty((num,(picks_per_it*(iterations)+initial_x)*(dim+1)))
    for i in range(num):
        print('On number: ' + str(i))
        BO_obj = myBO.BO(ub,lb,step,dim = dim,exp_w = exp_w,kernel = kernel,system = PM_H.BO_intializer,bounds = bnds,picks_per_level = picks_per_it)  
        BO_obj.optimizergp(iterations,1,num_xinit = initial_x) 
        #ys[i] = BO_obj.ygp
        #xs[i] = BO_obj.xgp.flatten()
        results[i,:(iterations*picks_per_it+initial_x)] = BO_obj.ygp.flatten()
        results[i,(iterations*picks_per_it+initial_x):] = BO_obj.xgp.flatten()
        ymax[i] = np.max(BO_obj.ygp)
        iteration[i] = int(np.argmax(BO_obj.ygp))
        xmax[i] = BO_obj.xgp[iteration[i]]
        xs_s[i] = BO_obj.xgp[0]
        ys_s[i] = BO_obj.ygp[0]
    
    '''
    un = np.unique(xs)
    print(max(ys))
    pyp.hist(xs,color = 'skyblue', bins = un)
    '''
    #headers = ['Biomass max','Light Intensity','Nitrogen','Qouta','light path','sun light hours']
    #df = pd.DataFrame(np.hstack([ymax,xmax]), columns = headers)
    df = pd.DataFrame(results)
    csv = "/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/High-D/BO/BO_dim_iterations_" + str(iterations + 1) + "_Repeated_" + str(num) + "_exploration_" + str(exp_w) + "_samplesperit_" + str(picks_per_it) + ".csv"
    df.to_csv(csv)

def high_dim_parallel_BO():
    '''
    Runs Bayesian Optimization for the algae model dependent upon light intensity, nitrogen, nitrogen qouta, sun light hours, and batch time
    '''

    ####Initializes Grid Creation
    
    #5 dimensions
    #Bounds
    bnds = []
    #Upper limit
    ub = np.array([2000,240,80,50,20])
    #Lower limit
    lb = np.array([0,0,20,10,12])
    #Step intervals
    step = np.array([80,40,10,5,1])
    #Dimensions
    dim = 5
    #Split up axis
    array_cons = np.array([5,2,2,2,1])
    
    start0 = time.time()
    time_results = []
    time_results.append(start0)
    
    '''
    #4 dimensions
    #Bounds
    bnds = []
    #Upper limit
    ub = np.array([2000,240,80,20])
    #Lower limit
    lb = np.array([0,0,20,12])
    #Step intervals
    step = np.array([80,40,10,1])
    #Dimensions
    dim = 4
    #Split up axis
    array_cons = np.array([5,2,2,2])
    '''
    '''
    #3 dimensions
    #Bounds
    bnds = []
    #Upper limit
    ub = np.array([2000,240,80])
    #Lower limit
    lb = np.array([0,0,20])
    #Step intervals
    step = np.array([80,20,2])
    #Dimensions
    dim = 3
    #Split up axis
    array_cons = np.array([5,4,2])
    '''
    '''
    #2 dimensions
    #Bounds
    bnds = []
    #Upper limit
    ub = np.array([2000,240])
    #Lower limit
    lb = np.array([0,0])
    #Step intervals
    step = np.array([20,4/3])
    #Dimensions
    dim = 2
    #Split up axis
    array_cons = np.array([10,4])
    '''

    fns = [PM_2,PM_3,PM_4,PM_5]
    num = 10
    xs = np.empty((num,dim))
    ys = np.empty((num,1))

    length_scs = (1,) * dim
    length_bnds = ((.1,10),)*dim

    kernel = gpr.kernels.Matern(length_scs, length_bnds, nu = 2.5)

    iterations = 5

    exp_w = 2.576
    initial_x = 10

    results = np.empty((num,((iterations)+1)*(dim+1)))

    #Finds consistent starting points
    from numpy.random import default_rng
    rng = default_rng(seed =  543)
    obj = myBO.BO(ub,lb,step,dim = dim,exp_w = exp_w,kernel = kernel,system = fns[dim - 2].BO_intializer,bounds = bnds)  
    obj.auto_create_axis()
    obj.iterate(0)

    initial_exps = [rng.choice(len(obj.grid_vertices), initial_x, replace=False) for _ in range(num)]

    time_results.append(time.time())

    for i in range(num):
        time_results.append(time.time())
        print('On number: ' + str(i))
        BO_obj = myBO.BO(ub,lb,step,dim = dim,exp_w = exp_w,kernel = kernel,system = fns[dim - 2].BO_intializer,bounds = bnds)  
        BO_obj.optimizersplt(iterations,np.prod(array_cons),[],array_cons,1,8,8,xinit = obj.grid_vertices[initial_exps[i],:])#num_xinit=initial_x)
        max_ind = np.argmax(BO_obj.yspltbst)
        results[i,:(iterations+1)] = BO_obj.yspltbst.flatten()
        results[i,(iterations+1):] = BO_obj.xspltbst.flatten()
        xs[i] = BO_obj.xspltbst[max_ind,:]

    time_results.append(time.time())

    print(xs)
    headers = ['Biomass max','Light Intensity','Nitrogen','Quota','Light Path','sun light hours']
    df = pd.DataFrame(results)
    csv = "/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/"+str(dim)+"-D/BO/BO_parallel_543_" + str(int(np.prod(array_cons))) + "_dim_" + str(dim) + "_iterations_" + str(iterations) + "_Repeated_" + str(num) + "_exploration_" + str(exp_w) + "_kernel.csv"
    df.to_csv(csv)

    ti = pd.DataFrame(time_results)
    ti.to_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/"+str(dim)+"-D/BO/BO_time_parallel_543_" + str(int(np.prod(array_cons))) + "_dim_" + str(dim) + "_iterations_" + str(iterations) + "_Repeated_" + str(num) + "_exploration_" + str(exp_w) + "_kernel.csv")
    
def high_dim_level_BO():
    '''
    Runs Bayesian Optimization for the algae model dependent upon light intensity, nitrogen, nitrogen qouta, sun light hours, and batch time
    '''
    ####Initializes Grid Creation
    '''
    #Bounds
    bnds = []
    #Upper limit
    ub = np.array([2000,240,80,50,20])
    #Lower limit
    lb = np.array([0,0,20,10,12])
    #Step intervals
    step = np.array([80,40,10,5,1])
    #Dimensions
    dim = 5
    #Split up axis
    array_cons = np.array([.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1])
    '''
    start0 = time.time()
    time_results = []
    time_results.append(start0)
    '''
    #4 Dimensions
    #Bounds
    bnds = []
    #Upper limit
    ub = np.array([2000,240,80,20])
    #Lower limit
    lb = np.array([0,0,20,12])
    #Step intervals
    step = np.array([80,40,10,1])
    #Dimensions
    dim = 4
    #Split up axis
    array_cons = np.array([.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1])
    '''
    '''
    #3 dimensions
    #Bounds
    bnds = []
    #Upper limit
    ub = np.array([2000,240,80])
    #Lower limit
    lb = np.array([0,0,20])
    #Step intervals
    step = np.array([80,20,2])
    #Dimensions
    dim = 3
    #Split up axis
    array_cons = np.array([.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1])
    '''
    
    #2 dimensions
    #Bounds
    bnds = []
    #Upper limit
    ub = np.array([2000,240])
    #Lower limit
    lb = np.array([0,0])
    #Step intervals
    step = np.array([20,4/3])
    #Dimensions
    dim = 2
    #Split up axis
    array_cons = np.array([.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1])
    

    fns = [PM_2,PM_3,PM_4,PM_5]

    num = 10
    xs = np.empty((num,dim))
    ys = np.empty((num,1))

    length_scs = (1,) * dim
    length_bnds = ((.1,10),)*dim

    kernel = gpr.kernels.Matern(length_scs, length_bnds, nu = 2.5)
    
    iterations = 5

    exp_w = 2.576
    initial_x = 10

    results = np.empty((num,((iterations)+1)*(dim+1)))

    #Finds consistent starting points
    from numpy.random import default_rng
    rng = default_rng(seed =  543)
    obj = myBO.BO(ub,lb,step,dim = dim,exp_w = exp_w,kernel = kernel,system = fns[dim-2].BO_intializer,bounds = bnds)  
    obj.auto_create_axis()
    obj.iterate(0)
    initial_exps = [rng.choice(len(obj.grid_vertices), initial_x, replace=False) for _ in range(num)]

    for i in range(num):
        time_results.append(time.time())
        print('On number: ' + str(i))
        BO_obj = myBO.BO(ub,lb,step,dim = dim,exp_w = exp_w,kernel = kernel,system = fns[dim-2].BO_intializer,bounds = bnds)  
        BO_obj.optimizerlevel(iterations,len(array_cons),array_cons,1,4,4,xinit = obj.grid_vertices[initial_exps[i]], picks_per_level=2)
        results[i,:(iterations+1)] = BO_obj.yspltbst.flatten()
        results[i,(iterations+1):] = BO_obj.xspltbst.flatten()

    time_results.append(time.time())

    headers = ['Biomass max','Light Intensity','Nitrogen','Quota','Light Path','sun light hours']
    df = pd.DataFrame(results)
    csv = "/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/"+str(dim)+"-D/BO/BO_level_seed_543_" + str(int(len(array_cons))) + "_dim_" + str(dim) + "_iterations_" + str(iterations) + "_Repeated_" + str(num) + "_exploration_" + str(exp_w) + "_kernel.csv"
    df.to_csv(csv)

    ti = pd.DataFrame(time_results)
    ti.to_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/"+str(dim)+"-D/BO/BO_time_level_543_" + str(int(np.prod(array_cons))) + "_dim_" + str(dim) + "_iterations_" + str(iterations) + "_Repeated_" + str(num) + "_exploration_" + str(exp_w) + "_kernel.csv")
    
def high_dim_splt_level():
    '''
    Runs Bayesian Optimization for the algae model dependent upon light intensity, nitrogen, nitrogen qouta, sun light hours, and batch time
    '''

    ####Initializes Grid Creation
    '''
    #5-D
    #Bounds
    bnds = []
    #Upper limit
    ub = np.array([2000,240,80,50,20])
    #Lower limit
    lb = np.array([0,0,20,10,12])
    #Step intervals
    step = np.array([80,40,10,5,1])
    #Dimensions
    dim = 5
    #Split up axis
    levels = np.array([1,.9,.8,.7])
    #Split up axis
    array_cons = np.array([5,1,1,2,1])
    '''
    start0 = time.time()
    time_results = []
    time_results.append(start0)
    
    '''
    #4-D
    #Bounds
    bnds = []
    #Upper limit
    ub = np.array([2000,240,80,20])
    #Lower limit
    lb = np.array([0,0,20,12])
    #Step intervals
    step = np.array([80,40,10,1])
    #Dimensions
    dim = 4
    #Split up axis
    levels = np.array([1,.9,.8,.7])
    #Split up axis
    array_cons = np.array([5,2,1,1])
    '''
    '''
    #3 dimensions
    #Bounds
    bnds = []
    #Upper limit
    ub = np.array([2000,240,80])
    #Lower limit
    lb = np.array([0,0,20])
    #Step intervals
    step = np.array([80,20,2])
    #Dimensions
    dim = 3
    #Split up axis
    levels = np.array([1,.9,.8,.7])
    #Split up axis
    array_cons = np.array([5,2,1])
    '''
    
    #2 dimensions
    #Bounds
    bnds = []
    #Upper limit
    ub = np.array([2000,240])
    #Lower limit
    lb = np.array([0,0])
    #Step intervals
    step = np.array([20,4/3])
    #Dimensions
    dim = 2
    #Split up axis
    levels = np.array([1,.9,.8,.7])
    #Split up axis
    array_cons = np.array([5,2])
    

    fns = [PM_2,PM_3,PM_4,PM_5]

    num = 10
    xs = np.empty((num,dim))
    ys = np.empty((num,1))

    length_scs = (1,) * dim
    length_bnds = ((.1,10),)*dim

    kernel = gpr.kernels.Matern(length_scs, length_bnds, nu = 2.5)

    iterations = 5

    exp_w = 2.576
    initial_x = 10

    results = np.empty((num,((iterations+1))*(dim+1)))

    #Finds consistent starting points
    from numpy.random import default_rng
    rng = default_rng(seed =  543)
    obj = myBO.BO(ub,lb,step,dim = dim,exp_w = exp_w,kernel = kernel,system = fns[dim-2].BO_intializer,bounds = bnds)  
    obj.auto_create_axis()
    obj.iterate(0)
    initial_exps = [rng.choice(len(obj.grid_vertices), initial_x, replace=False) for _ in range(num)]

    for i in range(num):
        time_results.append(time.time())
        print('On number: ' + str(i))
        BO_obj = myBO.BO(ub,lb,step,dim = dim,exp_w = exp_w,kernel = kernel,system = fns[dim-2].BO_intializer,bounds = bnds)  
        BO_obj.optimizerlevelsplt(iterations,len(array_cons),array_cons,levels,1,4,4,xinit = obj.grid_vertices[initial_exps[i]])
        results[i,:(iterations+1)] = BO_obj.yspltbst.flatten()
        results[i,(iterations+1):] = BO_obj.xspltbst.flatten()

    time_results.append(time.time())

    df = pd.DataFrame(results)
    csv = "/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/"+str(dim)+"-D/BO/BO_spltlevel_seed_543_" + str(int(np.prod(array_cons))) + "_dim_" + str(dim) + "_iterations_" + str(iterations) + "_Repeated_" + str(num) + "_exploration_" + str(exp_w) + "_kernel.csv"
    df.to_csv(csv)

    ti = pd.DataFrame(time_results)
    ti.to_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/"+str(dim)+"-D/BO/BO_time_spltlevel_seed_543_" + str(int(np.prod(array_cons))) + "_dim_" + str(dim) + "_iterations_" + str(iterations) + "_Repeated_" + str(num) + "_exploration_" + str(exp_w) + "_kernel.csv")
    
def high_dim_splt_dynamic():
    '''
    Runs Bayesian Optimization for the algae model dependent upon light intensity, nitrogen, nitrogen qouta, sun light hours, and batch time
    '''

    ####Initializes Grid Creation
    #Bounds
    bnds = []
    #Upper limit
    ub = np.array([2000,240,80,50,20])
    #Lower limit
    lb = np.array([0,0,20,10,12])
    #Step intervals
    step = np.array([80,40,10,5,1])
    #Dimensions
    dim = 5
    #Split up axis
    array_cons = np.array([5,2,2,2,1])

    length_scs = (1,) * dim
    length_bnds = ((.1,100),)*dim
    kernel = gpr.kernels.Matern(length_scs, length_bnds, nu = 2.5)

    intermediate_steps = [10]
    
    num = 10
    xs = np.empty((num,dim+len(intermediate_steps)))
    ys = np.empty((num,1))

    kernel = gpr.kernels.Matern(length_scs,length_bnds, nu = 2.5)

    iterations = 5

    exp_w = 2.576
    initial_x = 10

    results = np.empty((num,((iterations)+1)*(dim+1+len(intermediate_steps))))

    from numpy.random import default_rng
    rng = default_rng(seed =  543)
    
    obj = myBO.BO(ub,lb,step,dim = dim,exp_w = exp_w,kernel = kernel,system = PM_5.BO_intializer,bounds = bnds)  
    obj.auto_create_axis()
    obj.iterate(0)

    initial_exps = [rng.choice(len(obj.grid_vertices), initial_x, replace=False) for _ in range(num)]

    for i in range(num):
        print('On number: ' + str(i))
        BO_obj = myBO.BO(ub,lb,step,dim = dim,exp_w = exp_w,kernel = kernel,system = PM_5.BO_dynamic,bounds = bnds)  
        BO_obj.optimizersplt_dynamic(iterations,np.prod(array_cons),array_cons,1,intermediate_steps,8,8,xinit = obj.grid_vertices[initial_exps[i],:])#num_xinit=initial_x)
        max_ind = np.argmax(BO_obj.yspltbst)
        results[i,:(iterations+1)] = BO_obj.yspltbst.flatten()
        results[i,(iterations+1):] = BO_obj.xspltbst.flatten()
        xs[i] = BO_obj.xspltbst[max_ind,:]

    print(xs)
    headers = ['Biomass max','Light Intensity','Nitrogen','Quota','Light Path','sun light hours']
    df = pd.DataFrame(results)
    csv = "/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/5-D/BO/BO_paralleldynamic_improved_smallbounds_" + str(int(np.prod(array_cons))) + "_dim_" + str(dim) + "_iterations_" + str(iterations) + "_Repeated_" + str(num) + "_exploration_" + str(exp_w) + "_IntermediateSteps_" + str(len(intermediate_steps)) + "_steps_" + str(intermediate_steps) + ".csv"
    df.to_csv(csv)

def high_dim_arg_test():
    '''
    Tests Photo_high_dim with selected arguments. This method serves as a way to test the functionality of the model. light intensity, nitrogen, nitrogen qouta, light path dimension, sun light hours, and batch time
    '''

    test_array = np.array([120,	360,	44,	100,	12])

    soly,solt = PM_H.it(test_array)
    
    fig, axs = pyp.subplots(3)
    
    axs[0].scatter(solt,soly[0,:])
    axs[0].set_ylabel('Biomass')
    axs[1].scatter(solt,soly[1,:])
    axs[1].set_ylabel('N Concentration')
    axs[2].scatter(solt,soly[2,:])
    axs[2].set_ylabel('Quota')
    axs[2].set_xlabel('Time')
    print(PM_H.BO_intializer(test_array))
    pyp.show()

    ### Looks at sensitvity of the Biomass concentration to the light path (very sensitive)
    path = np.linspace(10,70,100)
    #path = np.linspace(10,24,13)
    y = []
    '''
    for i in range(len(path)):
        print(i)
        test_array[-3] = path[i]
        soly,solt = PM_H.it(test_array)
        y.append(soly[0,-1])
    
    pyp.scatter(path,y)
    pyp.show()
    '''

def Parallel_test():
    '''
    Testing BO.optimizersplt. 
    '''
    eqn = lambda x: np.sin(x[0]-2) + np.cos(1.3*x[1]-1.5)
    eqn1 = lambda x,y: np.sin(x-2) + np.cos(1.3*y-1.5)

    #Bounds
    bnds = Bounds((0,0),(1,1))
    #Upper limit
    ub = np.array([2*np.pi,2*np.pi])
    #Lower limit
    lb = np.array([0,0])
    #Step intervals
    step = np.array([np.pi/100,np.pi/100])
    #Dimesnions
    dim = 2
    #how dimensions will be split
    ar_cons = np.array([2,2])

    iterations = 15

    exp_w = 2.576

    kernel = gpr.kernels.Matern((1),(1,10), nu = 2.5)

    BO_obj = myBO.BO(ub,lb,step,dim = dim,exp_w = exp_w,kernel = kernel,system = eqn,bounds = bnds)  
    
    axis = create_axis(2,ub,lb,step)
    X, Y = np.meshgrid(axis[0], axis[1])
    Z = eqn1(X,Y)
    
    pyp.contour(X, Y, Z, levels=50)
    max_index = np.argmax(Z.flatten())

    max_x = X.flatten()[max_index]
    max_y = Y.flatten()[max_index]

    # Plot a dot at the maximum value
    pyp.plot(max_x, max_y, 'black')  # 'ro' specifies red color and circular marker
    pyp.text(max_x, max_y, f'Max ({max_x:.2f}, {max_y:.2f})', fontsize=10, ha='right')

    BO_obj.optimizersplt(iterations,0,[],ar_cons,1)
    print(BO_obj.grid_vertices)
    x = BO_obj.xsplt[:,0]
    y = BO_obj.xsplt[:,1]
    print(x)
    print(y)
    print(BO_obj.ysplt)
    pyp.scatter(x,y,color = 'red',zorder = 5)
    
    BO_obj.optimizergp_man(59,1,None)
    X1 = BO_obj.xgp[:,0]
    X2 = BO_obj.xgp[:,1]

    pyp.scatter(X1,X2,color = 'green',zorder = 4)
    pyp.title('Parallezied (red) vs Single Iteration (green) BO each with 60 samples')
    pyp.ylabel('X2')
    pyp.xlabel('X1')
    pyp.show()

#high_dim_splt_dynamic()
high_dim_splt_level()
#high_dim_basic_BO()
high_dim_level_BO()
high_dim_parallel_BO()
#Photo_N_arg_test()
#high_dim_arg_test()
#Parallel_test()