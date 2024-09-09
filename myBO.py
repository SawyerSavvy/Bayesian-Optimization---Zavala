import numpy as np
from matplotlib import pyplot as pyp, cm
from scipy.optimize import minimize, Bounds, approx_fprime
from scipy.misc import derivative
from joblib import Parallel, delayed
import sklearn.gaussian_process as gpr
from collections import OrderedDict
import torch
import time
import itertools
import random


class LCB_AF():
    """
    Lower Confidence Bound acquisition function capable
    of operating with and without a reference model
    """
    def __init__(self, model, dim, exp_w, descale, **refmod):
        '''
        Parameters:
            - model:
            - dim:
            - exp_w:
            - descale: 
            - refmod: This is the reference model
        '''
        self.model = model
        self.dim = dim
        self.exp_w = exp_w
        self.descale = descale
        #This batch size optimizes the time to calculate the uncertainity for large datasets
        self.batch = 1500000
        if refmod:
            self.refmod = refmod['refmod']
        else:
            def zr(x):
                return 0
            self.refmod = zr
    def LCB(self, x):
        '''
        Lower Confidence Bound

        This acquistiion function 
        '''
        #Creates a 1-D array with many rows
        #x = np.array([x]).reshape(-1,1)
        
        #Creates a matrix of dimesions of num/self.dim by self.dim
        #print(x)
        #print(x.shape)
        #x = x.reshape(int(x.shape[0]/self.dim), self.dim)
        
        # returns means & SD of predicitve distribution of all the query points 
        
        if len(x) > self.batch:
            #TODO: I should parallelize this
            l = []
            for i in range(int(np.ceil(len(x)/self.batch))):
                if i == np.ceil(len(x)/self.batch) - 1:
                    mu, std = self.model.predict(x[i*self.batch:], return_std=True)
                else:
                    mu, std = self.model.predict(x[i*self.batch:(i+1)*self.batch], return_std=True)
                res = (mu+2.576*std).flatten()
                l.append(res)

            return np.concatenate(l).flatten()
        
        else:
            mu, std = self.model.predict(x, return_std=True)
            #collasped into one dimension
            #mu = mu
            '''
            if str(type(self.refmod))=="<class '__main__.Network'>":
                yref = self.refmod(torch.from_numpy(x).float()).data.numpy()  
            else:
                yref = self.refmod(self.descale(x))
            '''
            
            #returns the uncertainity (look at equation in Acquisiton Functions)
            #pyp.plot(self.descale(x),yref+mu-self.exp_w*std.flatten())
            #pyp.plot(self.descale(x),mu, linestyle = 'dashed', color = 'red', label = 'Predicted f(x)')
            #pyp.fill_between(self.descale(x).ravel(),mu-self.exp_w*std,mu+self.exp_w*std,alpha = 0.2,color = 'red', label = '99% Confidence Interval')
            #return (yref+mu+self.exp_w*std).flatten()
            return (mu+self.exp_w*std).flatten()
    
class qLCB():
    """
    Multipoint LCB acquisition function
    """
    def __init__(self, model, q, dim, exp_w, samps):
        self.model = model
        self.q = q
        self.dim = dim
        self.exp_w = exp_w
        self.n = samps
    def LCB(self, x):
        x = x.reshape(self.q, self.dim)
        if np.unique(np.round(x, 4), axis = 0).shape[0]<self.q:
            return np.max(self.model.predict(x))
        else:
            mu, Sigma = self.model.predict(x, return_cov = True)
            L = np.linalg.cholesky(Sigma)
            S = 0
            for i in range(self.n):
                z = np.random.normal(np.zeros(mu.shape), np.ones(mu.shape), mu.shape)
                s = mu-self.exp_w*np.abs(L@z)
                S += np.min(s)
            S = S/(self.n)
            return S

class LCB_EMBD():
    """
    Work in progress
    """
    def __init__(self, model, var_num, dim, exp_w, fun, descale, include_x, **refmod):
        self.model = model
        self.var_num = var_num
        self.dim = dim
        self.exp_w = exp_w
        self.fun = fun
        self.descale = descale
        self.include_x = include_x
        if refmod:
            self.refmod = refmod['refmod']
        else:
            def zr(x):
                return np.zeros(x.shape)
            self.refmod = zr
    def LCB(self, x):
        x = x.reshape(-1, 1)
        x = x.reshape((int(x.shape[0]/self.dim), self.dim))
        mu = np.ones((x.shape[0], self.var_num))
        std = mu.copy()
        b = np.ones((x.shape[0], self.var_num+1))
        sigma = mu.copy()
        yref = self.refmod(self.descale(x))
        for i in range(self.var_num):
            mu[:, i], std[:, i] = self.model[str(i+1)].predict(x, return_std = True)
        eps = (yref+mu)*1e-3
        y0 = yref+mu+eps
        if self.include_x:
            y0 = np.hstack([y0, self.descale(x)])
        fp = np.ones(y0.shape)
        for i in range(x.shape[0]):
            fp[i] = approx_fprime(y0[i], self.fun, eps[i])
        b[:, 0] = self.fun(y0)
        b[:, 1:] = -fp*y0
        sigma = (fp**2*std**2)**0.5
        MU = np.sum(b, axis = 1)+np.sum(fp*mu, axis = 1)
        SIGMA = np.sum(sigma**2, axis = 1)**0.5
        return MU-self.exp_w*SIGMA

def partial_derivative(func, var=0, point=[]):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx = 1e-6)

class BO():
    '''
    Contains functions to execute Bayesian Optimization
    '''
    def __init__(self, ub, lb, step, dim, exp_w, kernel, system, bounds, **aux_mods):
        '''
        Initializes parameters for Bayseian Optimization
        Parameters:
            - ub: array of upperbound values for each dimension
            - lb: array of lowerbound values for each dimension
            - step: array of step intervals for each dimension
            - dim (int): number of dimensions
            - exp_w (float): exploration rate hyperparameter
            - system (function): objective function
            - bounds (scipy.optimize object): contains the descale boundaries for dimension. EX for 3 dimensions: scipy.optimize.Bounds((0,0,0),(1,1,1))
        '''

        #upper bound of design space
        self.ub = ub
        
        #lower bound of design space
        self.lb = lb

        #step size for design space
        self.step = step

        #Dimesions of x
        self.dim = dim

        #exploration rate
        self.exp_w = exp_w

        #Kernel
        self.kernel = kernel
        
        #objective function
        self.system = system

        #bounds of x (in scaled form)
        self.bounds = bounds

        self.dist_ref = {}

        if aux_mods:
            self.refmod = {'refmod': list(aux_mods.values())[0]}
            if len(aux_mods) > 1:
                self.distmod = aux_mods['distmod']
                self.dist_ref['distrefmod'] = aux_mods['ref_distmod']
                for i in range(3,len(aux_mods)):
                    self.dist_ref['distrefmod'+str(i-2)] = aux_mods['ref_distmod'+str(i-2)]

                self.dist_ref = OrderedDict(self.dist_ref)
                    
    def descale(self, x, scaling_factor = 1):
        '''
        Descales parameters from the scaling_factor (usually 0 to 1)
        Parameters:
            - x (array or float): values to be descaled
            - scaling_factor (float): number to be descaled from
        Returns: descaled values
        '''
        m = ((self.ub-self.lb)/scaling_factor)
        b = (self.lb)
        return m*x+b
    
    def scale(self, x, scaling_factor = 1):
        '''
        Scales parameters to the scaling_factor (usually 0 to 1)
        Parameters:
            - x (array or float): values to be scaled
            - scaling_factor (float): number to be scaled to
        Returns: scaled values
        '''
        m = scaling_factor/(self.ub-self.lb)
        b = -self.lb/(self.ub-self.lb)*scaling_factor
        return m*x+b
    
    def random_xinit_generator(self,sf,num_xinit):
        '''
        ##TODO: no longer used
        Generates a matrix of random number that are in the specified grid at from 0 to the scaling factor
        Parameters:
            - sf:           scaling factor
            - num_xinit:    number of random points within the grid to be generated
        returns:            numpy matrix of random x-values in the grid
        '''

        full_axis = np.empty((self.dim,),dtype=object)

        #upper scale of each axis
        ub_scale = self.scale(self.ub)
        #lower scale of each axis
        lb_scale = self.scale(self.lb)
        #step size of each axis
        step_scale = self.step / (self.ub - self.lb) * sf

        for i in range(self.dim):
            number_steps = int(np.ceil((ub_scale[i] - lb_scale[i])/step_scale[i]))
            full_axis[i] = np.linspace(lb_scale[i],ub_scale[i],num = number_steps+1)

        ran_xinit = np.empty((num_xinit,self.dim),dtype = object)

        for j in range(num_xinit):
            for i in range(len(full_axis)):
                ran_xinit[j,i] = np.random.choice(full_axis[i,:])

        return ran_xinit

    def auto_create_axis(self):
        '''
        Creates a list of axis for every grid. 
        It has no arguments
            - Creates axis from upper bounds, lower bounds, and step lists. A axis is created for each dimension/index in the list
        Returns nothing. The full list is stored as a instance variable called self.axis
        '''
        #upper scale of each axis
        ub_scale = self.scale(self.ub)
        #lower scale of each axis
        lb_scale = self.scale(self.lb)
        #step size of each axis
        step_scale = self.step / (self.ub - self.lb)

        #intializes axis 
        full_axis = np.empty((self.dim,),dtype=object)
        
        #creates the axis for the full grid
        for i in range(self.dim):
            number_steps = int(np.ceil((ub_scale[i] - lb_scale[i])/step_scale[i]))
            full_axis[i] = np.linspace(lb_scale[i],ub_scale[i],num = number_steps+1)
            #full_axis[i] = np.arange(lb_scale[i],ub_scale[i]+step_scale[i],step_scale[i])

        # Checks to see if the subspace will be parallelized
        try:
            if self.ar_cons is None: 
                self.axis = full_axis
                return
        except AttributeError:
            self.axis = full_axis
            return
        
        #Splits the axis into separate sections
        for i in range(self.dim):
            num_grids = self.ar_cons[i]
            len_axis = len(full_axis[i])

            #if number of desired grids is equal to or larger the number of elements, each element will be defined as a seperate grid
            if num_grids >= len_axis:
                axis_spl = np.reshape(full_axis[i],(len(full_axis),1))

            else:
                #If merge is true, the remainder will merge with the last section of the split axis
                merge = True

                #Remainder of the elements of the axis divded by the number of desired sections
                remain = len_axis % num_grids

                #If remainder is greater than half of the length of the sections, don't merge
                if remain > 0.5 * (len(full_axis[i]))/num_grids:
                    merge = False
                
                #Deals with the remainder
                if remain > 0:
                    axis_spl = np.split(full_axis[i][:-remain],num_grids)

                    if merge:
                        #Adds to the last section
                        axis_spl[-1] = np.append(axis_spl[-1],[full_axis[i][-remain:]])
                    
                    else:
                        #creates a new section
                        axis_spl = np.append(axis_spl,full_axis[i][-remain:])
                
                else:
                    axis_spl = np.split(full_axis[i],num_grids)

            #Assigns the split axis
            full_axis[i] = axis_spl

        #The split axis
        self.axis = full_axis

        return

    def man_grid(self,manual_grid):
        '''
        Creates a grids for each partition based on the manually inputted list. 
        Self.split is the number of partitions within the design space. 
        return: if # of partitions equals 1, then return the manual_grid as a list
                if # of partitions > 1, then return an array that holds each partition's list of vertices
        '''
        if(self.split == 1):
            #scales all x-dimensions
            self.grid_vertices = np.array(manual_grid)
            self.grid_vertices[:,:self.dim] = self.scale(self.grid_vertices[:,:self.dim])
            print(self.grid_vertices)
            return
        
        #finds number of elements per split
        num_grid = len(manual_grid) / self.split

        #assigns a grid to each partiton
        for num in range(self.split):
            #for last partition, assigns the remaining data
            if (num == self.split - 1):
                self.grid_vertices[-1] = manual_grid[num*num_grid:]
                break
            
            #each row receives a grid, num*num_grid
            self.grid_vertices[num] = manual_grid[num * num_grid: (num + 1) * num_grid - 1]

        return

    def create_axis(self, low_bd, high_bd, index = -1):
        '''
            TODO: This is incorrect, use auto_create_axis instead
            Creates every axis for each dimension by starting from lower to upper increasing by step for every index.
            ---- self.axis is a matrix that for each row contains the values of each dimension
        '''
        # should be -1 when not calling from LS-BO
        if (index == -1):
            ub_scale = self.scale(self.ub)
            lb_scale = self.scale(self.lb)
            step_scale = self.step / (self.ub - self.lb)
            
            # for when there is only 1 dimension
            if(isinstance(step_scale,float)): 
                self.axis = np.arange(lb_scale,ub_scale+step_scale,step_scale)
                return
            
            # for when dimensions > 1
            #intialize matrix
            self.axis = np.empty((self.dim,),dtype=object)
            
            #assign each row a list a dimension's values
            for s in range(self.dim):
                self.axis[s] = np.arange(lb_scale[s],ub_scale[s]+step_scale[s],step_scale[s])
        
            self.axis[index] = np.empty((self.dim,),dtype=object)
            
        #Applys to LS-BO
        else:
            ub_scale = self.scale(high_bd)
            lb_scale = self.scale(low_bd)
            step_scale = self.step / (self.ub-self.lb)

            # for when there is only 1 dimension
            if(isinstance(step_scale,float)): 
                self.axis[index] = np.arange(lb_scale,ub_scale+step_scale,step_scale)
                return
            
            # for when dimensions > 1
            self.axis[index] = np.empty((self.dim,),dtype=object)
            
            for s in range(self.dim):
                self.axis[index][s] = np.arange(low_bd[s],high_bd[s]+step_scale[s],step_scale[s])
        return
    
    def iterate(self, parallel = -1):
        '''
            Creates a grid with all possible vertices from the axis matrix
            If parallelel, it recursively identifies the axis for each grid
            Arguments:
                - Parallel: if 0, then will not create multiple grids for parallelization
            Return: 
                - if not parallel, returns an array of vertices
                - if parallel, returns an list of matrices, where each matrix holds the vertices of an grid
        '''
        # applys to everything except LS-BO
        if (parallel == 0):
            #for when dimensions < 1
            if self.dim == 1:
                self.grid_vertices = self.axis.reshape(-1,1)
                return
            self.grid_vertices = np.array(list(itertools.product(*self.axis)))

        # applys to LS-BO
        else:
            # for dimensions equal to 1
            if self.dim == 1:
                for row in range(len(self.axis)):
                    self.grid_vertices[row] = self.axis[row]
                    return
            
            vectors = [0]*self.dim
            self.grid_vertices = self.recursion_iterate(self.axis,vectors,0,[])

        return
    
    def recursion_iterate(self,matrix,vectors,current_row,all_grids):
        '''
            Recursively iterates through the matrix generated in auto_create_axis to make a list of grids to be evaluated in parallel

            Parameters:
                - matrix: a matrix whose rows contains all split sections of an axis
                - vectors: current axis for the grid that will be generated
                - current_row: current row of the matrix
                - all_grids: a list of grids that have been made so far

            Returns:
                - all_grids: a list of all possible grids
        '''
        if current_row == len(matrix):

            all_grids.append(np.array(list(itertools.product(*vectors))))
            return

        for v in range(len(matrix[current_row])):

            vectors[current_row] = matrix[current_row][v]
            self.recursion_iterate(matrix,vectors,current_row+1,all_grids)
        
        return all_grids
    
    def optimizergp(self, trials, scaling_factor, cores = 4, xinit = None, yinit = None, num_xinit = 1, picks_per_level = 1):
        """
        Standard Bayesian optimization algorithm run using the LCB acquisition function.
        The 'cores' argument sets the number of cores that will be used for AF optimization
        """
        print('Vanilla BO Run...')
        start = time.time()

        self.ar_cons = None
        self.trialsgp = trials
        
        self.timegp = np.ones(self.trialsgp+1)

        # Time for each trial held in an array
        self.timefgp = np.ones(self.trialsgp+1)

        sf = scaling_factor
                
        self.auto_create_axis()

        self.iterate(parallel = 0)

        if xinit is None:
            #uniformly draw samples
            y = np.empty((num_xinit,)).reshape(-1,1)
            x = np.random.uniform(0, sf, (num_xinit, self.dim))
        
            # limits the random numbers vertices on the grid
            step_scale = self.step / (self.ub - self.lb)
            x = np.around(x / step_scale) * step_scale

        else:
            #1-D array with self.dim elements
            x = self.scale(xinit).reshape(-1, self.dim)
            y = np.empty((self.dim,x.shape[1]),dtype=object)
        
        startf = time.time()

        #objective function which outputs y values of the x-values
        if yinit is None:
            for i in range(x.shape[0]):
                y[i] = self.system(self.descale(x[i]))
        else:
            y = yinit

        endf = time.time()
        
        #collected how much time passed to calculate output of objective function
        self.timefgp[0] = endf-startf

        #creating gaussian process object
        modelgp = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6, normalize_y = True, n_restarts_optimizer = 10)

        #Fit the gaussian process to the data
        modelgp.fit(x, y)

        #array that contains the uncertainity of the points
        LCBgp = LCB_AF(modelgp, self.dim, self.exp_w, self.descale).LCB

        # using multiple parallel cpus for the function minimize which is analyzing the LCBgp function
        uncert = LCBgp(self.grid_vertices)

        xnxt = np.zeros((picks_per_level,self.dim))
        ynxt = np.zeros((picks_per_level,1))

        for i in range(self.trialsgp):
            
            for z in range(picks_per_level):
                    # LCB minimizing
                    opt = np.argmax(uncert)
                    #opt = Parallel(n_jobs = afcores)(delayed(np.argmin)(uncert[uncert > (self.cons[i]*ran + maxi)]))
                    
                    #finds minimum for x-values
                    xnxt[z] = uncert[opt]

                    uncert[opt] = -np.inf

                    #next y-value that is evaluated
                    ynxt[z] = np.array([self.system(self.descale(xnxt[z]))])

            #evaluating the time it takes to computes y-value
            self.timefgp[i+1] = self.timefgp[i]+(endf-startf)
            
            #stacking x & xnxt
            x = np.vstack([x, xnxt])
            
            #stacking y & ynxt
            y = np.vstack([y, ynxt])

            #updating gaussian process
            modelgp.fit(x, y)

            #results of the optimzation for x values
            uncert = LCBgp(self.grid_vertices)
        
        self.modelgp = modelgp

        #x-values for gaussian process
        self.xgp = self.descale(x)
        
        #y-values collected
        self.ygp = y

        self.ref_optim = False
        self.splt_optim = False
        self.splt_optimref = False
        self.spltvar_optim = False
        self.hyp_optim = False
        self.expwl_optim = False
        self.nmc_optim = False
        self.qBO_optim = False
        self.embd_optim = False

    def optimizergp_man(self, trials, scaling_factor, grid, xinit = None, yinit = None):
        """
        Standard Bayesian optimization algorithm run using the LCB acquisition function. This method takes the average of all outputs found at the same argument
        The 'cores' argument sets the number of cores that will be used for AF optimization
        Parameters:
            - self: instance of the class
            - trials (int): number of iterations of updating the gaussian model (note that the total number of BO rounds is trials + 1)
            - scaling_factor (int): scaling factor to scale all values to (usually 1)
            - grid (int): numpy array of all possible vertices
            - cores (int): not currently used
            - xinit (if dim = 1, float numpy array, else float numpy matrix): contains the known x-values
            - yinit (if dim = 1, float numpy array, else float numpy matrix): contains the known y-values
        """
        print('Vanilla BO Run with man grid...')
        start = time.time()
        self.trialsgp = trials
        
        self.timegp = np.ones(self.trialsgp+1)

        # Time for each trial held in an array
        self.timefgp = np.ones(self.trialsgp+1)
        
        #Only one point analyzed at a time. 
        self.split = 1

        self.ar_cons = None
        
        if grid is not None:
            self.grid_vertices = self.scale(grid,scaling_factor)
        else:
            self.auto_create_axis()
            self.iterate(parallel=0)

        #no intial x-values
        if xinit is None:

            #uniformly draw samples
            y = np.empty((1)).reshape(-1,1)

            #Generates a random number which represents the row number to be used
            x_index = np.random.randint(0, len(self.grid_vertices)-1)
            x = self.grid_vertices[x_index,:self.dim].reshape(1,-1)
            y[0] = self.system(self.descale(x[-1]))

        #intial x-values
        else:
            #1-D array with self.dim elements
            x = self.scale(xinit).reshape(-1, self.dim)
            #TODO: this may cause me trouble
            y = np.empty((x.shape),dtype=float)
            #objective function which outputs y values of the x-values
            if yinit is None:
                for i in range(len(x)):
                    y[i] = self.system(self.descale(x))
            else:
                y = yinit

        startf = time.time()

        y = y.reshape(-1,1)

        endf = time.time()
        
        #collected how much time passed to calculate output of objective function
        self.timefgp[0] = endf-startf

        #creating gaussian process object
        modelgp = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6, normalize_y = True, n_restarts_optimizer = 10)

        #Fit the gaussian process to the data
        modelgp.fit(x, y)

        #creates the acquisition model object
        LCBgp = LCB_AF(modelgp, self.dim, self.exp_w, self.descale).LCB

        #Calculates the mean + uncertainty via the acquisition function
        uncert = LCBgp(self.grid_vertices)

        # calculating time for acquisition function calculation
        end = time.time()
        self.timegp[0] = end-start

        #val = np.linspace(self.lb, self.ub,1001)
        for i in range(self.trialsgp):
            '''
            if (i != 4):
                pyp.plot(val,self.system(val),color = 'black',label = 'f(x)')
                pyp.scatter(xinit,yinit,color = 'black',label = 'Initial Data')
                pyp.scatter(self.descale(x),y,color = 'blue', label = 'Data')
                if (i != 0):
                    pyp.scatter(self.descale(x[-1]),y[-1],color = 'green', marker = 's')
                pyp.ylim(bottom = 0)
                pyp.xlim(self.lb[0],self.ub[0])
                pyp.title('n = ' + str(len(x)))
                #pyp.legend()
            '''

            #next x-value that is evaluated, at the maximum acquistion function value
            xnxt = self.grid_vertices[np.argmax(uncert),:]

            startf = time.time()
                
            #next y-value that is evaluated
            ynxt = self.system(self.descale(xnxt))

            #evaluating the time it takes to computes y-value
            endf = time.time()
            self.timefgp[i+1] = self.timefgp[i]+(endf-startf)
            
            #stacking x & xnxt
            x = np.vstack([x, xnxt])
            
            #stacking y & ynxt
            y = np.vstack([y, ynxt])

            #updating gaussian process
            modelgp.fit(x, y)

            '''
            if i != 2:
                pyp.subplot(3,1,i+2)
            else:
                pyp.subplots_adjust(wspace=0.5, hspace=0.5)
                pyp.show()
            '''
            #results of the optimzation for x values
            uncert = LCBgp(self.grid_vertices)

            end = time.time()
            self.timegp[i+1] = end-start
        
        self.modelgp = modelgp

        #x-values for gaussian process
        self.xgp = self.descale(x)
        
        #y-values collected
        self.ygp = y

        self.ref_optim = False
        self.splt_optim = False
        self.splt_optimref = False
        self.spltvar_optim = False
        self.hyp_optim = False
        self.expwl_optim = False
        self.nmc_optim = False
        self.qBO_optim = False
        self.embd_optim = False
    
    def optimizergp_man_SD(self, trials, scaling_factor, grid, cores = 4, xinit = None, yinit = None):
        """
        Standard Bayesian optimization algorithm run using the LCB acquisition function. This factors the variation of multiple experiments at the same arguments.
        The 'cores' argument sets the number of cores that will be used for AF optimization
        """
        
        print('Vanilla BO Run with man grid...')
        start = time.time()
        self.trialsgp = trials
        
        self.timegp = np.ones(self.trialsgp+1)

        # Time for each trial held in an array
        self.timefgp = np.ones(self.trialsgp+1)

        sf = scaling_factor
        
        #Only one point analyzed at a time. 
        self.split = 1

        #creates grid that will be analyzed
        self.man_grid(grid)
        y_grid = grid[:,4]

        self.xgp = []
        self.ygp = []
        x = []
        y = []
        print(grid)
        #no intial x-values
        if xinit is None:
            #uniformly draw samples
            y = np.empty((1)).reshape(-1,1)

            #Generates a random number which represents the row number to be used
            #TODO: if random number has multiple same vertices, need to average y-values
            x_index = np.random.randint(0, len(grid))
            x_one = self.grid_vertices[x_index,:self.dim].reshape(1,-1)
            print(self.grid_vertices)
            print(y_grid)
            y_ar = self.system(x_one,np.hstack((self.grid_vertices,y_grid.reshape(-1,1))))
            x = np.tile(x_one, (len(y_ar),1))
            y = y_ar
            self.xgp.append(np.unique(x_one,axis = 0))
            self.ygp.append(np.unique(y_ar,axis=0))
            

        #intial x-values
        else:
            #1-D array with self.dim elements
            x_vals = self.scale(xinit).reshape(-1, self.dim)

            #objective function which outputs y values of the x-values
            for i in range(x.shape[0]):
                y_ar = self.system(x_vals[i],self.grid_vertices)
                x.append(np.tile(x_vals[i],(len(y_ar),1)))

        startf = time.time()

        y = y.reshape(-1,1)

        endf = time.time()
        
        #collected how much time passed to calculate output of objective function
        self.timefgp[0] = endf-startf

        #creating gaussian process object
        modelgp = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6, normalize_y = True, n_restarts_optimizer = 10)

        #Fit the gaussian process to the data
        modelgp.fit(x, y)

        #array that contains the uncertainity of the points
        LCBgp = LCB_AF(modelgp, self.dim, self.exp_w, self.descale).LCB

        # using multiple parallel cpus for the function minimize which is analyzing the LCBgp function
        #pyp.subplot(3,1,1)
        uncert = LCBgp(self.grid_vertices[:,:self.dim])


        # calculating time for acquisition function calculation
        
        end = time.time()
        self.timegp[0] = end-start
        val = np.linspace(self.lb, self.ub,1001)
        for i in range(self.trialsgp):
            '''
            if (i != 4):
                pyp.plot(val,self.system(val),color = 'black',label = 'f(x)')
                pyp.scatter(xinit,yinit,color = 'black',label = 'Initial Data')
                pyp.scatter(self.descale(x),y,color = 'blue', label = 'Data')
                if (i != 0):
                    pyp.scatter(self.descale(x[-1]),y[-1],color = 'green', marker = 's')
                pyp.ylim(bottom = 0)
                pyp.xlim(self.lb[0],self.ub[0])
                pyp.title('n = ' + str(len(x)))
                #pyp.legend()
            '''

            #next x-value that is evaluated, at the minimum acquistion function value
            xnxt = self.grid_vertices[np.where(uncert == np.max(uncert)),:self.dim].reshape(-1,self.dim)
            
            #TODO: to be removed during paralleziation and actual DOE. 
            #  The reason this exists is because I don't want already evaluated experiments to be used again which will lower standard deviation and mess with kernel function
            if (np.any(np.all(self.xgp == xnxt, axis =1))):
                #skip the loop
                self.xgp.append(xnxt)
                self.ygp.append(self.ygp[np.where(self.xgp == xnxt)])
                continue

            startf = time.time()

            #next y-value that is evaluated
            ynxt = np.array([self.system(xnxt,np.hstack((self.grid_vertices,y_grid)))])

            xnxt = np.tile(xnxt,(len(ynxt),1))

            #evaluating the time it takes to computes y-value
            endf = time.time()
            self.timefgp[i+1] = self.timefgp[i]+(endf-startf)
            
            #stacking x & xnxt
            x = np.vstack([x, xnxt])
            
            #stacking y & ynxt
            y = np.vstack([y, ynxt])

            #updating gaussian process
            modelgp.fit(x, y)

            #randomly collects a distribution of numbers between 0 & sf in the matrix of 128 by self.dim
            #x0 = np.random.uniform(0, sf, (128, self.dim))
            # using multiple parallel cpus for the function minimize which is analyzing the LCBgp function
            #opt = Parallel(n_jobs = cores)(delayed(minimize)(LCBgp, x0 = start_point,method = 'L-BFGS-B',bounds = self.bounds)for start_point in x0)
            '''
            if i != 2:
                pyp.subplot(3,1,i+2)
            else:
                pyp.subplots_adjust(wspace=0.5, hspace=0.5)
                pyp.show()
            '''
            #results of the optimzation for x values
            uncert = LCBgp(self.grid_vertices[:,:4])

            end = time.time()
            self.timegp[i+1] = end-start
        
        self.modelgp = modelgp

        #x-values for gaussian process
        self.xgp = self.descale(x)
        
        #y-values collected
        self.ygp = y

        self.ref_optim = False
        self.splt_optim = False
        self.splt_optimref = False
        self.spltvar_optim = False
        self.hyp_optim = False
        self.expwl_optim = False
        self.nmc_optim = False
        self.qBO_optim = False
        self.embd_optim = False    

    def optimizeref(self, trials, scaling_factor, cores = 4, xinit = None):
        """
        BO with a reference model algorithm as described in Q. Lu et al. The 
        reference model is introduced as an entry in the **aux_mods dictionary
        with the key 'refmod'. The 'cores' argument sets the number of cores
        that will be used for AF optimization
        """
        print('BO with Reference Model Run...')
        start = time.time()
        self.trialsref = trials
        self.timeref = np.ones(self.trialsref+1)
        self.timefref = np.ones(self.trialsref+1)
        refmod = self.refmod['refmod']
        sf = scaling_factor

        # 
        if xinit is None:
            x = np.random.uniform(0, sf, (1, self.dim))
        else:
            x = xinit.reshape(1, self.dim)
        startf = time.time()

        self.create_axis()
        self.iterate()

        y = self.system(self.descale(x))
        if str(type(refmod))=="<class '__main__.Network'>":
            eps = y - refmod(torch.from_numpy(x).float()).data.numpy()
        else:
            eps = y - refmod(self.descale(x))
        endf = time.time()
        self.timefref[0] = endf-startf

        #creates the gaussian process
        self.modeleps = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6, 
                                                  normalize_y = True,
                                                  n_restarts_optimizer = 10)
        #fits it to data
        self.modeleps.fit(x.reshape(1,self.dim), eps)

        #creates a bunch of random points
        x0 = np.random.uniform(0, sf, (128, self.dim))

        #creates an array of the uncertainites of each x-value
        LCBref = LCB_AF(self.modeleps, self.dim, self.exp_w, self.descale, **self.refmod).LCB

        #minimizes the uncertainites
        #opt = Parallel(n_jobs = cores)(delayed(minimize)(LCBref, x0 = start_point,method = 'L-BFGS-B', tol = 1e-6,bounds = self.bounds)for start_point in x0)

        uncert = LCBgp(self.grid_vertices)

        #Finds the optimal solution
        #xnxts = np.array([res.x for res in opt], dtype  = 'float')

        #finds the y-value of the x-values
        #funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
        
        end = time.time()
        self.timeref[0] = end-start

        for i in range(self.trialsref):
            #computes the next x-value from finding the min value in funs
            #xnxt = xnxts[np.argmin(funs)].reshape(1, self.dim)
            xnxt = self.grid_vertices[np.where(uncert == np.min(uncert))]
            startf = time.time()

            #computes the y-value from the optimized x-value
            ynxt = self.system(self.descale(xnxt))

            if str(type(refmod))=="<class '__main__.Network'>":
                epsnxt = ynxt - refmod(torch.from_numpy(xnxt).float()).data.numpy()
            else:
                epsnxt = ynxt - refmod(self.descale(xnxt))
            endf = time.time()

            self.timefref[i+1] = self.timefref[i]+(endf-startf)

            #joining the new x-value into the x value set
            x = np.vstack([x, xnxt])

            #joining the new y-value in the y value set
            y = np.vstack([y, ynxt])

            # this has something related to the model
            eps = np.vstack([eps, epsnxt])

            
            self.modeleps.fit(x, eps)

            #more points
            x0 = np.random.uniform(0, sf, (128, self.dim))

            #computing acquisiion function
            #opt = Parallel(n_jobs = cores)(delayed(minimize)(LCBref, x0 = start_point,method = 'L-BFGS-B', tol = 1e-6,bounds = self.bounds)for start_point in x0)
            
            #find the potentials
            uncert = LCBgp(self.grid_vertices)
            #finding optimization
            #xnxts = np.array([res.x for res in opt], dtype  = 'float')

            #finding y-values
            #funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            
            end = time.time()
            self.timeref[i+1] = end-start
        
        self.ref_optim = True
        self.xref = self.descale(x)
        self.ytru = y
        self.eps = eps

    def optimizerlevel(self, trials, partition_num, partition_cons, scaling_factor, fcores = 4, afcores = 4, xinit = None, num_xinit = 1, man_grid = None,picks_per_level = 1):
        """
        LS-BO approach developed by us and explained in González, L.D. et al. partition_cons
        should be a dictionary of constraints numerically indexed and contained in lists that
        set up the level-set partitions of the feature space according to whatever partition
        shape is desired; bounds can be linear or nonlinear; base these off reference model
        In this function, reference model is not used to guide the search as in the following.
        The 'fcores' argument sets the number of cores used for parallel experimentation, and
        the 'afcores' argument sets the cores used for optimizing the AF

        Parameters:
            trials: the iterations of BO
            partition_num: the number of grids
            partition_cons: the constrains on each grid
            scaling_factor: factor to scale from 0 to 1 when fitting to GP
            fcores: cores used in paralleizing function analysis
            afcores: cores used in finding acquisition function values
            xinit: array of inital x-values that already are known
        """
        print('Partitioned Domain BO Run...')
        start = time.time()
        self.trialsplt = trials
        self.split = partition_num
        self.cons = partition_cons
        self.timesplt = np.ones(self.trialsplt+1)
        self.timefsplt = np.ones(self.trialsplt+1)
        sf = scaling_factor
        x = np.array([]).reshape(0, self.dim)
        
        if man_grid is None:
            self.auto_create_axis()
            #Develops the grid from the axis
            self.iterate(parallel = 0)

        #Randomly generates x-values
        if xinit is None:
            x = np.random.uniform(0, sf, (num_xinit, self.dim))

            #limits x-values to the values on the grid
            step_scale = self.step / (self.ub - self.lb)
            x = np.around(x / step_scale) * step_scale
        else:
            x = xinit.reshape(-1,self.dim)

        # objective function is being used, finds y-value for each x-value
        y = Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point)) for start_point in x)
        
        #stacks the y-values
        y = np.hstack(y[:]).T.reshape(-1,1)

        #Finds best y-value
        xbst = x[np.argmax(y)]
        ybest = max(y).reshape(-1,1)

        #fitting the gaussian process
        modelsplt = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,
                                                 n_restarts_optimizer = 10,
                                                 normalize_y = True)
        
        #fitting model to data
        begin = time.time()
        modelsplt.fit(x, y)
        end = time.time()
        print('fit ' + str(end-begin))

        # Array for holding next x-values
        xnxt = np.ones((self.split*picks_per_level, self.dim))

        #uncertainintes from AQ function
        #intialize acquisition function
        LCBgp = LCB_AF(modelsplt, self.dim, self.exp_w, self.descale).LCB

        #uncert = Parallel(n_jobs = afcores)(delayed(LCB)(self.grid_vertices))
        begin = time.time()
        uncert = LCBgp(self.grid_vertices)
        end = time.time()
        print('predict ' + str(end-begin))
        
        #results = Parallel(n_jobs = afcores)(delayed(LCBgp)(grid.reshape(1,-1)) for grid in self.grid_vertices)

        mini = np.min(uncert)
        maxi = np.max(uncert)
        ran = maxi - mini

        #For each grid, maximize the acquistion function start starting at the start points, find the predicted best x-values, finds the maximum y-value of the best x-values
        for i in range(self.split):
            
            for z in range(picks_per_level):
                microgrid = uncert[uncert <= (self.cons[i]*ran + mini)]
                # LCB minimizing
                opt = np.argmax(microgrid)
                #opt = Parallel(n_jobs = afcores)(delayed(np.argmin)(uncert[uncert > (self.cons[i]*ran + maxi)]))
                
                #finds minimum for x-values
                xnxt[i*picks_per_level+z] = self.grid_vertices[opt]

                microgrid[opt] = -np.inf

            

        #For the number of trials
        for i in range(self.trialsplt):
            #system calculation
            #Calculates the y-value for each
            ynxt =  Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point)) for start_point in xnxt)
            #ynxt =  Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point)) for start_point in xnxtbs)
            
            ynxt = np.hstack(ynxt[:]).T.reshape(-1, 1)
            
            #adding new data points to data set
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt])
            
            #Updates best y & x value
            if np.max(ynxt) > np.max(ybest):
                ybest = np.vstack([ybest, np.max(ynxt)])
                xbst = np.vstack([xbst, xnxt[np.argmax(ynxt),:]])
            elif xbst.ndim == 1:
                ybest = np.vstack([ybest, np.max(ybest)])
                xbst = np.vstack([xbst, xbst])
            else:
                ybest = np.vstack([ybest, np.max(ybest)])
                xbst = np.vstack([xbst, xbst[-1,:]])
            
            #fitting guassian process to new data
            #Fit to the data
            modelsplt.fit(x, y)
            
            #uncert = Parallel(n_jobs = afcores)(delayed(LCB)(self.grid_vertices))
            uncert = LCBgp(self.grid_vertices)
            #results = Parallel(n_jobs = afcores)(delayed(LCB)(grid) for grid in self.grid_vertices)

            mini = np.min(uncert)
            maxi = np.max(uncert)
            ran = maxi - mini

            #For each grid, maximize the acquistion function start starting at the start points, find the predicted best x-values, finds the minimum y-value of the best x-values
            for j in range(self.split):
                '''
                # LCB minimizing
                opt = np.argmax(uncert[uncert < (self.cons[j]*ran + mini)])
                #opt = Parallel(n_jobs = afcores)(delayed(np.argmax)(uncert <= self.cons[i*ran + maxi]))
                
                #finds minimum for x-values
                xnxt[j] = self.grid_vertices[opt]
                '''
                for z in range(picks_per_level):
                    microgrid = uncert[uncert <= (self.cons[j]*ran + mini)]
                    # LCB minimizing
                    opt = np.argmax(microgrid)
                    #opt = Parallel(n_jobs = afcores)(delayed(np.argmin)(uncert[uncert > (self.cons[i]*ran + maxi)]))
                    
                    #finds minimum for x-values
                    xnxt[j*picks_per_level+z] = self.grid_vertices[opt]

                    microgrid[opt] = -np.inf
                    end = time.time()
                
            self.timesplt[i+1] = end-start

        self.splt_optim = True
        self.modelsplt = modelsplt
        self.xsplt = self.descale(x)
        self.ysplt = y
        self.yspltbst = ybest
        self.xspltbst = self.descale(xbst)
   
    def optimizerlevelsplt(self, trials, partition_num, partition_cons, levels, scaling_factor, fcores = 4, afcores = 4, xinit = None, num_xinit = 1, yinit = None, manual_grid = None,picks_per_level = 1):
        """
        Grid split into blocks

        Parameters:
            - Trials:           loops of Bayesian Optimization for each grid
            - Partition_num:    the number of grids that will be analyzed within the design space
            - Partition_cons:    an array of integers indicating how the axis should be split up to make grids
                                    e.g. x = [1,2,4] ----> this means the first axis won't be split. The second axis will be split in half. The third axis will be split 4 times. 
                                        Overall, this means the number of grids is 1*2*4 = 8. 
            - level_cons:       a list of floats between 0 and 1 stating how the level sets of each partiton should be split up
            - scaling_factor:   What value will the data be scaled down to. 
            - The 'fcores' argument sets the number of cores used for parallel experimentation
            - 'afcores' argument sets the cores used for optimizing the AF
            - xinit: array containing what x-values train the surrogate model first 
            - yinit: array containing the y-values corresponding to the x-values in xinit
            - manual_grid: import the grid that will be analyzed by this function. This should already be split into a list where each element holds a separate grid
        """

       
        # partition_cons is an array designating how you want the axis of each grid split up
        #partition_num splits the grid into designated amount of grids by spliting the grid array into the desingnated number of grids

        #partition_cons contains the constraints for the minimize function
        #how to put in partition_cons
        print('Partitioned Domain BO Run...')

        start = time.time()
        
        #number of trials
        self.trialsplt = trials
        #lists of partition constraints
        self.levels = levels
        #numberical values of constraints
        self.ar_cons = partition_cons

        self.timesplt = np.ones(self.trialsplt+1)
        self.timefsplt = np.ones(self.trialsplt+1)
        
        sf = scaling_factor

        #empty array with self.dim elements 
        x = np.array([]).reshape(0, self.dim)

        # if manual_grid is intialized skip generation of matrix
        if manual_grid is None:
            #Creates the axis for each grid
            self.auto_create_axis()
            #Develops the grid from the axis
            self.iterate()

        else:
            self.split = partition_num
            self.man_grid(manual_grid)

        #determines how many grids there are
        self.split = len(self.grid_vertices)

        # repeats based on number of paritions
        if xinit is None:
            x = np.random.uniform(0, sf, (num_xinit, self.dim))

            #limits x-values to the values on the grid
            step_scale = self.step / (self.ub - self.lb)
            x = np.around(x / step_scale) * step_scale

        else:
            x = xinit.reshape(-1,self.dim)
        
        y = Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point)) for start_point in x)
    
        startf = time.time()
        endf = time.time()
        self.timefsplt[0] = endf-startf
        
        #reshaping y
        y = np.hstack(y[:]).reshape(-1,1)

        #Finds maximum y-value for the intial points
        ybest = max(y).reshape(-1,1)
        
        #makes a list of best x-values in chronological order
        xbst = x[np.argmax(y),:]
        
        #fitting the gaussian process
        modelsplt = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,n_restarts_optimizer = 10,normalize_y = True)
        
        modelsplt.fit(x, y)

        #creating first matrix of recorded x-values for each split
        xnxt = np.ones((self.split*len(self.levels)*picks_per_level, self.dim))

        #Creates acquisiiton model object
        LCB = LCB_AF(modelsplt, self.dim, self.exp_w, self.descale).LCB

        results = Parallel(n_jobs = afcores)(delayed(LCB)(grid) for grid in self.grid_vertices)

        for i in range(self.split):
            mini = np.min(results[i])
            maxi = np.max(results[i])
            ran = maxi - mini
            for j in range(len(self.levels)):
                #finds maximum for x-values
                for z in range(picks_per_level):
                    microgrid = results[i][results[i] <= (self.levels[j]*ran + mini)]
                    top_arg = np.argmax(microgrid)
                    xnxt[i*len(self.levels)*picks_per_level+j*picks_per_level+z] = self.grid_vertices[i][top_arg]
                    microgrid[top_arg] = -np.inf
        
        end = time.time()
        self.timesplt[0] = end-start

        #Total trials to be run
        for i in range(self.trialsplt):
            
            startf = time.time()
        
            #Calculates the y-value for the x-values (one in each grid)
            ynxt = np.array(Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point)) for start_point in xnxt))
            
            endf = time.time()
            self.timefsplt[i+1] = self.timefsplt[i]+(endf-startf)
            
            #adding new data points to data set
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt.reshape(-1,1)])
            
            #Updates best y & x value
            if np.max(ynxt) > np.max(ybest):
                ybest = np.vstack([ybest, np.max(ynxt)])
                xbst = np.vstack([xbst, xnxt[np.argmax(ynxt),:]])
            elif xbst.ndim == 1:
                ybest = np.vstack([ybest, np.max(ybest)])
                xbst = np.vstack([xbst, xbst])
            else:
                ybest = np.vstack([ybest, np.max(ybest)])
                xbst = np.vstack([xbst, xbst[-1,:]])

            modelsplt.fit(x, y)

            #for each split, finds the max for the acquisiton function
            results = Parallel(n_jobs = afcores)(delayed(LCB)(grid) for grid in self.grid_vertices)

            for a in range(self.split):
                mini = np.min(results[a])
                maxi = np.max(results[a])
                ran = maxi - mini
                for j in range(len(self.levels)):
                    #finds maximum for x-values
                    for z in range(picks_per_level):
                        microgrid = results[a][results[a] <= (self.levels[j]*ran + mini)]
                        top_arg = np.argmax(microgrid)
                        xnxt[a*len(self.levels)*picks_per_level+j*picks_per_level+z] = self.grid_vertices[a][top_arg]
                        microgrid[top_arg] = -np.inf

            end = time.time()
            
            self.timesplt[i+1] = end-start

        self.splt_optim = True
        self.modelsplt = modelsplt
        self.xsplt = self.descale(x)
        self.ysplt = y
        self.yspltbst = ybest
        self.xspltbst = self.descale(xbst)

   #######
   # LS-BO 
   #     
    def optimizersplt(self, trials, partition_num, partition_cons, array_cons, scaling_factor, fcores = 4, afcores = 4, xinit = None, yinit = None, manual_grid = None,num_xinit = 20):
        """
        Argument Selections are identified from grids breaking up the domain space. Only one Guassian Process is trained across all data

        Parameters:
            - Trials:           loops of Bayesian Optimization for each grid
            - Partition_num:    the number of grids that will be analyzed within the design space
            - Partition_cons:   TODO: not used currently
                                An matrix containing the proportions of each axis that you want as a grid. This must be scaled and be a square. Meaning each axis must have equal splits
                                    e.g.  x = [[[0, .25],[0, .25],[0, .25]],[[.25,.75],[.25,.75],[.25,.75]],[[.75, 1],[.75,1],[.75,1]]] 
            - array_cons:       an array of integers indicating how the axis should be split up to make grids
                                    e.g. x = [1,2,4] ----> this means the first axis won't be split. The second axis will be split in half. The third axis will be split 4 times. 
                                        Overall, this means the number of grids is 1*2*4 = 8. 
            - scaling_factor:   What value will the data be scaled down to. 
            - The 'fcores' argument sets the number of cores used for parallel experimentation
            - 'afcores' argument sets the cores used for optimizing the AF
            - xinit: array containing what x-values train the surrogate model first 
            - yinit: array containing the y-values corresponding to the x-values in xinit
            - manual_grid: import the grid that will be analyzed by this function. This should already be split into a list where each element holds a separate grid
        """

       
        # partition_cons is an array designating how you want the axis of each grid split up
        #partition_num splits the grid into designated amount of grids by spliting the grid array into the desingnated number of grids

        #partition_cons contains the constraints for the minimize function
        #how to put in partition_cons
        print('Partitioned Domain BO Run...')

        start = time.time()
        
        #number of trials
        self.trialsplt = trials
        #lists of partition constraints
        self.cons = partition_cons
        #numberical values of constraints
        self.ar_cons = array_cons

        self.timesplt = np.ones(self.trialsplt+1)
        self.timefsplt = np.ones(self.trialsplt+1)
        
        sf = scaling_factor

        # if manual_grid is intialized skip generation of matrix
        if manual_grid is None:
            #Creates the axis for each grid
            self.auto_create_axis()
            #Develops the grid from the axis
            self.iterate()
            #determines how many grids there are
            self.split = len(self.grid_vertices)
        else:
            self.split = partition_num
            self.man_grid(manual_grid)
            #updates self.split in case partition_num > len(self.grid_verticies)
            self.split = len(self.grid_vertices)

        # repeats based on number of paritions
        if xinit is None:
            x = np.random.uniform(0,sf, (num_xinit, self.dim))

            # limits the random numbers vertices on the grid
            step_scale = self.step / (self.ub - self.lb)
            x = np.around(x / step_scale) * step_scale

            # Assigns each fcore to a specific row of xbs that is evaluated in self.system
            y = Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point)) for start_point in x)

        else:
            x = xinit.reshape(-1,self.dim)
            y = Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point)) for start_point in x)
        
        #reshaping y
        y = np.hstack(y[:]).reshape(-1,1)

        #Finds maximum y-value for the intial points
        ybest = max(y).reshape(-1,1)
        
        #makes a list of best x-values in chronological order
        xbst = x[np.argmax(y)]
        
        #fitting the gaussian process
        modelsplt = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,n_restarts_optimizer = 10,normalize_y = True)
        
        #fitting model to data
        modelsplt.fit(x, y)

        #creating first matrix of recorded x-values for each split
        xnxt = np.ones((self.split, self.dim))

        #Creates acquisition model object
        LCB = LCB_AF(modelsplt, self.dim, self.exp_w, self.descale).LCB

        results = Parallel(n_jobs = afcores)(delayed(LCB)(grid) for grid in self.grid_vertices)

        for i in range(self.split):
            #finds maximum for x-values
            xnxt[i] = self.grid_vertices[i][np.argmax(results[i])]

        #Total trials to be run
        for i in range(self.trialsplt):
            
            startf = time.time()
        
            #Calculates the y-value for the x-values (one in each grid)
            ynxt = np.array(Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point)) for start_point in xnxt))
            
            #adding new data points to data set
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt.reshape(-1,1)])
            
            #Updates best y & x value
            if np.max(ynxt) > np.max(ybest):
                ybest = np.vstack([ybest, np.max(ynxt)])
                xbst = np.vstack([xbst, xnxt[np.argmax(ynxt),:]])
            elif xbst.ndim == 1:
                ybest = np.vstack([ybest, np.max(ybest)])
                xbst = np.vstack([xbst, xbst])
            else:
                ybest = np.vstack([ybest, np.max(ybest)])
                xbst = np.vstack([xbst, xbst[-1,:]])

            #fitting guassian process to new data
            modelsplt.fit(x, y)

            #for each split, finds the max for the acquisiton function
            results = Parallel(n_jobs = afcores)(delayed(LCB)(grid) for grid in self.grid_vertices)

            for j in range(self.split):
                xnxt[j] = self.grid_vertices[j][np.argmax(results[j])]

            if i == self.trialsplt - 1:
                print(modelsplt.kernel_.length_scale)

        self.splt_optim = True
        self.modelsplt = modelsplt
        self.xsplt = self.descale(x)
        self.ysplt = y
        self.yspltbst = ybest
        self.xspltbst = self.descale(xbst)


    def optimizersplt_dynamic(self, trials, partition_num, array_cons, scaling_factor, intermediate_list, fcores = 4, afcores = 4, xinit = None, yinit = None, manual_grid = None,num_xinit = 20):
        """
        Argument Selections are identified from grids breaking up the domain space. A nested gaussian process is used to facilitate training using dynamic data. 

        Parameters:
            - Trials:           loops of Bayesian Optimization for each grid
            - Partition_num:    the number of grids that will be analyzed within the design space
            - array_cons:       an array of integers indicating how the axis should be split up to make grids
                                    e.g. x = [1,2,4] ----> this means the first axis won't be split. The second axis will be split in half. The third axis will be split 4 times. 
                                        Overall, this means the number of grids is 1*2*4 = 8. 
            - scaling_factor:   What value will the data be scaled down to. 
            - intermediate_steps: a list of the time_intervals that should be evaluated
            - The 'fcores' argument sets the number of cores used for parallel experimentation
            - 'afcores' argument sets the cores used for optimizing the AF
            - xinit: array containing what x-values train the surrogate model first 
            - yinit: array containing the y-values corresponding to the x-values in xinit
            - manual_grid: import the grid that will be analyzed by this function. This should already be split into a list where each element holds a separate grid
        """
        def scale_inter(x,max,min):
            '''
            Scales parameters to the scaling_factor (usually 0 to 1)
            Parameters:
                - x (array or float): values to be scaled
                - scaling_factor (float): number to be scaled to
            Returns: scaled values
            '''

            m = 1/(max-min)
            b = -min/(max-min)*1

            return m*x+b

        def dynamic_data_gp(time_steps, train_x, conditions, modelsplt, n = 0): 
            '''
            This function is used for the nested GPs in prediciting Biomass at each time step. This is called for every intermediate time step that is used. 
            
            Parameters: 
                - n: counts number of iterations
                - time_steps: used in recursion
                - conditions: list of arguments
                - train_x: argument training data -> [arg1, arg2, ..., argN, output_1, output_2, ..., output_M-1] where M is the number of time steps
                - returns: matrix with intermidate time interval values assigned at every arugment -> [arg1, arg2, ..., argN, output_1, output_2, ..., output_M-1]
            '''

            def predict_all(model,x):
                batch = self.batch = 1500000
                if len(x) > batch:
                    #TODO: I should parallelize this
                    l = []
                    for i in range(int(np.ceil(len(x)/batch))):
                        if i == np.ceil(len(x)/self.batch) - 1:
                            mu, std = model.predict(x[i*batch:], return_std=True)
                        else:
                            mu, std = model.predict(x[i*batch:(i+1)*batch], return_std=True)
                        res = mu.flatten()
                        l.append(res)

                    return np.concatenate(l).flatten()
                
                else:
                    mu, std = model.predict(x, return_std=True)
                    return mu
            
            length_scs = (1,) * (self.dim+n)
            length_bnds = ((.1,1000),)* (self.dim+n)
            ker = gpr.kernels.Matern(length_scs, length_bnds, nu = 2.5)
            modelsplt = gpr.GaussianProcessRegressor(ker, alpha = 1e-6,n_restarts_optimizer = 10,normalize_y = True)

            #Train GP
            modelsplt.fit(train_x[:,:(self.dim+n)],train_x[:,(self.dim+n)])

            pa = predict_all

            #Output Aqusition fn for every condition
            #conditions[:][:][self.dim+n] = Parallel(n_jobs = afcores)(delayed(LCB)(grid) for grid in conditions)
            outputs = Parallel(n_jobs = afcores)(delayed(pa)(modelsplt,grid[:,:self.dim+n]) for grid in conditions)

            large = np.max([np.max(arr) for arr in outputs])
            small = np.min([np.min(arr) for arr in outputs])

            for i in range(len(outputs)):
                conditions[i][:,self.dim+n] = scale_inter(outputs[i],large,small)

            #Final step in Recursion should be at the last intermediate time_interval
            if n == time_steps - 1: 
                return conditions
            else:
                return dynamic_data_gp(time_steps, train_x, conditions, modelsplt,n+1)
       
        # partition_cons is an array designating how you want the axis of each grid split up
        #partition_num splits the grid into designated amount of grids by spliting the grid array into the desingnated number of grids

        #partition_cons contains the constraints for the minimize function
        #how to put in partition_cons
        print('Partitioned Domain BO Run...')

        start = time.time()
        
        #number of trials
        self.trialsplt = trials
        #numberical values of constraints
        self.ar_cons = array_cons

        self.timesplt = np.ones(self.trialsplt+1)
        self.timefsplt = np.ones(self.trialsplt+1)
        
        sf = scaling_factor
        intermediate_steps = len(intermediate_list)

        # if manual_grid is intialized skip generation of matrix
        if manual_grid is None:
            #Creates the axis for each grid
            self.auto_create_axis()
            #Develops the grid from the axis
            self.iterate()

            for i in range(len(self.grid_vertices)):
                self.grid_vertices[i] = np.hstack([self.grid_vertices[i],np.zeros((len(self.grid_vertices[i]),intermediate_steps))])

            #determines how many grids there are
            self.split = len(self.grid_vertices)
        else:
            self.split = partition_num
            self.man_grid(manual_grid)
            #updates self.split in case partition_num > len(self.grid_verticies)
            self.split = len(self.grid_vertices)

        # repeats based on number of paritions
        if xinit is None:
            x = np.random.uniform(0,sf, (num_xinit, self.dim))

            # limits the random numbers vertices on the grid
            step_scale = self.step / (self.ub - self.lb)
            x = np.around(x / step_scale) * step_scale

            #Adds matrix of zeros representing the known intermediate values
            x = np.hstack([x,np.zeros((len(x),intermediate_steps))])

            # Assigns each fcore to a specific row of xbs that is evaluated in self.system
            y_full = np.array(Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point[:self.dim])) for start_point in x))
       
            y = y_full[:,0,-1]
          
            x[:,self.dim:] = y_full[:,0,intermediate_list].reshape(-1, intermediate_steps)

        else:

            x = xinit

            step_scale = self.step / (self.ub - self.lb)
            x = np.around(x / step_scale) * step_scale
            #Adds matrix of zeros representing the known intermediate values
            x = np.hstack([x,np.zeros((len(x),intermediate_steps))])

            # Assigns each fcore to a specific row of xbs that is evaluated in self.system
            y_full = np.array(Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point[:self.dim])) for start_point in x))
       
            y = y_full[:,0,-1]
          
            x[:,self.dim:] = y_full[:,0,intermediate_list].reshape(-1, intermediate_steps)
        
        #reshaping y
        y = np.hstack(y[:]).reshape(-1,1)

        #Finds maximum y-value for the intial points
        ybest = max(y).reshape(-1,1)
        
        #makes a list of best x-values in chronological order
        xbst = x[np.argmax(y)]
        
        #fitting the gaussian process
        modelsplt = []#gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,n_restarts_optimizer = 10,normalize_y = True)
        
        self.grid_vertices = dynamic_data_gp(intermediate_steps,x,self.grid_vertices,modelsplt)

        ma = np.max(x[:,self.dim:],axis=0)
        mi = np.min(x[:,self.dim:],axis=0)
        #fitting model to data

        length_scs = (1,) * (self.dim+intermediate_steps)
        length_bnds = ((.1,1000),)* (self.dim+intermediate_steps)
        ker = gpr.kernels.Matern(length_scs, length_bnds, nu = 2.5)
        modelsplt = gpr.GaussianProcessRegressor(ker, alpha = 1e-6,n_restarts_optimizer = 10,normalize_y = True)

        modelsplt.fit(np.hstack([x[:,:self.dim],scale_inter(x[:,self.dim:],ma,mi)]), y)
        print(np.hstack([x[:,:self.dim],scale_inter(x[:,self.dim:],ma,mi)]))
        print(modelsplt.kernel_.length_scale)
        #creating first matrix of recorded x-values for each split
        xnxt = np.ones((self.split, self.dim+intermediate_steps))
        #Creates acquisition model object
        LCB = LCB_AF(modelsplt, self.dim, self.exp_w, self.descale).LCB

        results = Parallel(n_jobs = afcores)(delayed(LCB)(grid) for grid in self.grid_vertices)

        

        for i in range(self.split):
            #finds maximum for x-values
            xnxt[i] = self.grid_vertices[i][np.argmax(results[i])]

        #Total trials to be run
        for i in range(self.trialsplt):
            
            startf = time.time()
        
            #Calculates the y-value for the x-values (one in each grid)
            y_full = np.array(Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point[:self.dim])) for start_point in xnxt))

            ynxt = y_full[:,0,-1]
            
            xnxt[:,self.dim:] = y_full[:,0,intermediate_list].reshape(-1, intermediate_steps)
            
            #adding new data points to data set
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt.reshape(-1,1)])

            #Updates best y & x value
            if np.max(ynxt) > np.max(ybest):
                ybest = np.vstack([ybest, np.max(ynxt)])
                xbst = np.vstack([xbst, xnxt[np.argmax(ynxt),:]])
            elif xbst.ndim == 1:
                ybest = np.vstack([ybest, np.max(ybest)])
                xbst = np.vstack([xbst, xbst])
            else:
                ybest = np.vstack([ybest, np.max(ybest)])
                xbst = np.vstack([xbst, xbst[-1,:]])

            self.grid_vertices = dynamic_data_gp(intermediate_steps,x,self.grid_vertices,modelsplt)

            ma = np.max(x[:,self.dim:],axis=0)
            mi = np.min(x[:,self.dim:],axis=0)

            #fitting guassian process to new data
            modelsplt.fit(np.hstack([x[:,:self.dim],scale_inter(x[:,self.dim:],ma,mi)]), y)

            #for each split, finds the max for the acquisiton function
            results = Parallel(n_jobs = afcores)(delayed(LCB)(grid) for grid in self.grid_vertices)

            for j in range(self.split):
                xnxt[j] = self.grid_vertices[j][np.argmax(results[j])]

            
            if i == self.trialsplt - 1:
                print(modelsplt.kernel_.length_scale)
                #quit()
            

        self.splt_optim = True
        self.modelsplt = modelsplt
        self.xsplt = np.hstack([self.descale(x[:,:self.dim]),x[:,self.dim:]])
        self.ysplt = y
        self.yspltbst = ybest
        self.xspltbst = np.hstack([self.descale(xbst[:,:self.dim]),xbst[:,self.dim:]])

    def optimizersplt_dynamic_simple(self, trials, partition_num, array_cons, scaling_factor, intermediate_list, fcores = 4, afcores = 4, xinit = None, yinit = None, manual_grid = None,num_xinit = 20):
        """
        Argument Selections are identified from grids breaking up the domain space. A nested gaussian process is used to facilitate training using dynamic data. 

        Parameters:
            - Trials:           loops of Bayesian Optimization for each grid
            - Partition_num:    the number of grids that will be analyzed within the design space
            - array_cons:       an array of integers indicating how the axis should be split up to make grids
                                    e.g. x = [1,2,4] ----> this means the first axis won't be split. The second axis will be split in half. The third axis will be split 4 times. 
                                        Overall, this means the number of grids is 1*2*4 = 8. 
            - scaling_factor:   What value will the data be scaled down to. 
            - intermediate_steps: a list of the time_intervals that should be evaluated
            - The 'fcores' argument sets the number of cores used for parallel experimentation
            - 'afcores' argument sets the cores used for optimizing the AF
            - xinit: array containing what x-values train the surrogate model first 
            - yinit: array containing the y-values corresponding to the x-values in xinit
            - manual_grid: import the grid that will be analyzed by this function. This should already be split into a list where each element holds a separate grid
        """
        def scale_inter(x,max,min):
            '''
            Scales parameters to the scaling_factor (usually 0 to 1)
            Parameters:
                - x (array or float): values to be scaled
                - scaling_factor (float): number to be scaled to
            Returns: scaled values
            '''

            m = 1/(max-min)
            b = -min/(max-min)*1

            return m*x+b

        def dynamic_data_gp(time_steps, train_x, conditions, modelsplt, n = 0): 
            '''
            This function is used for the nested GPs in prediciting Biomass at each time step. This is called for every intermediate time step that is used. 
            
            Parameters: 
                - n: counts number of iterations
                - time_steps: used in recursion
                - conditions: list of arguments
                - train_x: argument training data -> [arg1, arg2, ..., argN, output_1, output_2, ..., output_M-1] where M is the number of time steps
                - returns: matrix with intermidate time interval values assigned at every arugment -> [arg1, arg2, ..., argN, output_1, output_2, ..., output_M-1]
            '''

            def predict_all(model,x):
                batch = self.batch = 1500000
                if len(x) > batch:
                    #TODO: I should parallelize this
                    l = []
                    for i in range(int(np.ceil(len(x)/batch))):
                        if i == np.ceil(len(x)/self.batch) - 1:
                            mu, std = model.predict(x[i*batch:], return_std=True)
                        else:
                            mu, std = model.predict(x[i*batch:(i+1)*batch], return_std=True)
                        res = mu.flatten()
                        l.append(res)

                    return np.concatenate(l).flatten()
                
                else:
                    mu, std = model.predict(x, return_std=True)
                    return mu
                
            length_scs = (1,)
            length_bnds = ((.1,100),)
            ker = gpr.kernels.Matern(length_scs, length_bnds, nu = 2.5)
            modelsplt = gpr.GaussianProcessRegressor(ker, alpha = 1e-6,n_restarts_optimizer = 10,normalize_y = True)

            #Train GP
            modelsplt.fit(train_x[:,(self.dim+n-1)].reshape(-1,1),train_x[:,(self.dim+n)].reshape(-1,1))

            pa = predict_all

            #Output Aqusition fn for every condition
            #conditions[:][:][self.dim+n] = Parallel(n_jobs = afcores)(delayed(LCB)(grid) for grid in conditions)
            outputs = Parallel(n_jobs = afcores)(delayed(pa)(modelsplt,grid[:,self.dim+n-1].reshape(-1,1)) for grid in conditions)

            large = np.max([np.max(arr) for arr in outputs])
            small = np.min([np.min(arr) for arr in outputs])

            for i in range(len(outputs)):
                conditions[i][:,self.dim+n] = scale_inter(outputs[i],large,small)

            #Final step in Recursion should be at the last intermediate time_interval
            if n == time_steps - 1: 
                return conditions
            else:
                return dynamic_data_gp(time_steps, train_x, conditions, modelsplt,n+1)
       
        # partition_cons is an array designating how you want the axis of each grid split up
        #partition_num splits the grid into designated amount of grids by spliting the grid array into the desingnated number of grids

        #partition_cons contains the constraints for the minimize function
        #how to put in partition_cons
        print('Partitioned Domain BO Run...')

        start = time.time()
        
        
        #number of trials
        self.trialsplt = trials
        #numberical values of constraints
        self.ar_cons = array_cons

        self.timesplt = np.ones(self.trialsplt+1)
        self.timefsplt = np.ones(self.trialsplt+1)
        
        sf = scaling_factor
        intermediate_steps = len(intermediate_list)

        # if manual_grid is intialized skip generation of matrix
        if manual_grid is None:
            #Creates the axis for each grid
            self.auto_create_axis()
            #Develops the grid from the axis
            self.iterate()

            for i in range(len(self.grid_vertices)):
                self.grid_vertices[i] = np.hstack([self.grid_vertices[i],np.zeros((len(self.grid_vertices[i]),intermediate_steps))])

            #determines how many grids there are
            self.split = len(self.grid_vertices)
        else:
            self.split = partition_num
            self.man_grid(manual_grid)
            #updates self.split in case partition_num > len(self.grid_verticies)
            self.split = len(self.grid_vertices)

        # repeats based on number of paritions
        if xinit is None:
            x = np.random.uniform(0,sf, (num_xinit, self.dim))

            # limits the random numbers vertices on the grid
            step_scale = self.step / (self.ub - self.lb)
            x = np.around(x / step_scale) * step_scale

            #Adds matrix of zeros representing the known intermediate values
            x = np.hstack([x,np.zeros((len(x),intermediate_steps))])

            # Assigns each fcore to a specific row of xbs that is evaluated in self.system
            y_full = np.array(Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point[:self.dim])) for start_point in x))
       
            y = y_full[:,0,-1]
          
            x[:,self.dim:] = y_full[:,0,intermediate_list].reshape(-1, intermediate_steps)

        else:

            x = xinit

            step_scale = self.step / (self.ub - self.lb)
            x = np.around(x / step_scale) * step_scale
            #Adds matrix of zeros representing the known intermediate values
            x = np.hstack([x,np.zeros((len(x),intermediate_steps))])

            # Assigns each fcore to a specific row of xbs that is evaluated in self.system
            y_full = np.array(Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point[:self.dim])) for start_point in x))
       
            y = y_full[:,0,-1]
          
            x[:,self.dim:] = y_full[:,0,intermediate_list].reshape(-1, intermediate_steps)
        
        #reshaping y
        y = np.hstack(y[:]).reshape(-1,1)

        #Finds maximum y-value for the intial points
        ybest = max(y).reshape(-1,1)
        
        #makes a list of best x-values in chronological order
        xbst = x[np.argmax(y)]
        
        #fitting the gaussian process
        modelsplt = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,n_restarts_optimizer = 10,normalize_y = True)
        
        self.grid_vertices = dynamic_data_gp(intermediate_steps,x,self.grid_vertices,modelsplt)

        ma = np.max(x[:,-1],axis=0)
        mi = np.min(x[:,-1],axis=0)

        length_scs = (1,1,1,1,1,1)
        length_bnds = ((.1,100),(.1,100),(.1,100),(.1,100),(.1,100),(.1,100))
        ker = gpr.kernels.Matern(length_scs, length_bnds, nu = 2.5)
        modelsplt = gpr.GaussianProcessRegressor(ker, alpha = 1e-6,n_restarts_optimizer = 10,normalize_y = True)

        #fitting model to data
        modelsplt.fit(np.hstack([x[:,:self.dim],scale_inter(x[:,-1].reshape(-1,1),ma,mi)]), y)

        #creating first matrix of recorded x-values for each split
        xnxt = np.ones((self.split, self.dim+intermediate_steps))

        #Creates acquisition model object
        LCB = LCB_AF(modelsplt, self.dim, self.exp_w, self.descale).LCB

        results = Parallel(n_jobs = afcores)(delayed(LCB)(grid[:,[0,1,2,3,4,-1]].reshape(-1,6)) for grid in self.grid_vertices)

        for i in range(self.split):
            #finds maximum for x-values
            xnxt[i] = self.grid_vertices[i][np.argmax(results[i])]

        #Total trials to be run
        for i in range(self.trialsplt):
            
            startf = time.time()
        
            #Calculates the y-value for the x-values (one in each grid)
            y_full = np.array(Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point[:self.dim])) for start_point in xnxt))

            ynxt = y_full[:,0,-1]
            
            xnxt[:,self.dim:] = y_full[:,0,intermediate_list].reshape(-1, intermediate_steps)
            
            #adding new data points to data set
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt.reshape(-1,1)])

            #Updates best y & x value
            if np.max(ynxt) > np.max(ybest):
                ybest = np.vstack([ybest, np.max(ynxt)])
                xbst = np.vstack([xbst, xnxt[np.argmax(ynxt),:]])
            elif xbst.ndim == 1:
                ybest = np.vstack([ybest, np.max(ybest)])
                xbst = np.vstack([xbst, xbst])
            else:
                ybest = np.vstack([ybest, np.max(ybest)])
                xbst = np.vstack([xbst, xbst[-1,:]])

            self.grid_vertices = dynamic_data_gp(intermediate_steps,x,self.grid_vertices,modelsplt)

            ma = np.max(x[:,-1],axis=0)
            mi = np.min(x[:,-1],axis=0)

            #fitting guassian process to new data
            modelsplt.fit(np.hstack([x[:,:self.dim],scale_inter(x[:,-1].reshape(-1,1),ma,mi)]), y)

            #for each split, finds the max for the acquisiton function
            results = Parallel(n_jobs = afcores)(delayed(LCB)(grid[:,[0,1,2,3,4,-1]].reshape(-1,6)) for grid in self.grid_vertices)

            for j in range(self.split):
                xnxt[j] = self.grid_vertices[j][np.argmax(results[j])]
            '''
            if i == self.trialsplt - 1:
                print(modelsplt.kernel_.length_scale)
                quit()
            '''

        self.splt_optim = True
        self.modelsplt = modelsplt
        self.xsplt = np.hstack([self.descale(x[:,:self.dim]),x[:,self.dim:]])
        self.ysplt = y
        self.yspltbst = ybest
        self.xspltbst = np.hstack([self.descale(xbst[:,:self.dim]),xbst[:,self.dim:]])

   #######
   # LS-BO but with a asymmetric model
   #     
    def optimizersplt_man(self, trials, num_max, scaling_factor, fcores = 4, afcores = 4, xinit = None, yinit = None, manual_grid = None):
        # partition_cons could be an array of upper and lower bounds x = [[[0, .25],[0, .25],[0, .25]],[[.25,.75],[.25,.75],[.25,.75]],[[.75, 1],[.75,1],[.75,1]]] 
        """
        LS-BO approach developed by us and explained in González, L.D. et al. partition_cons
        should be a dictionary of constraints numerically indexed and contained in lists that
        set up the level-set partitions of the feature space according to whatever partition
        shape is desired; bounds can be linear or nonlinear; base these off reference model
        In this function, reference model is not used to guide the search as in the following.
        The 'fcores' argument sets the number of cores used for parallel experimentation, and
        the 'afcores' argument sets the cores used for optimizing the AF
        """
        #partition_cons contains the constraints for the minimize function
        #how to put in partition_cons
        print('Partitioned Domain BO Run...')

        start = time.time()
        
        #number of trials
        self.trialsplt = trials

        self.timesplt = np.ones(self.trialsplt+1)
        self.timefsplt = np.ones(self.trialsplt+1)
        
        sf = scaling_factor

        #empty array with self.dim elements 
        x = np.array([]).reshape(0, self.dim)
        switch = True
        
        #added systems
        self.axis = np.empty((self.split,),dtype = object)
        self.grid_vertices = np.empty((self.split),dtype = object)

        # if manual_grid is intialized skip generation of matrix
        if manual_grid is None:
            self.auto_create_axis()
            self.iterate(parallel = 1)
        else:
            self.man_grid(manual_grid)

        #In summary, the code performs a loop over partitions, checks if a given initial point satisfies the 
        # constraints within the partition, and updates the initial point accordingly. If the initial point 
        # does not satisfy the constraints, it generates random initial points and performs optimization to 
        # find the optimal point within the constraints.

        # repeats based on number of paritions
        if xinit is None:
            for i in range(self.split):
                x0 = np.random.uniform(self.grid_vertices[i][0],self.grid_vertices[i][-1], (1, self.dim))
                x = np.vstack([x, x0.reshape(1, self.dim)])
            # Assigns each fcore to a specific row of xbs that is evaluated in self.system
            y = Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point)) for start_point in x)

        else:
            x = xinit
            y = yinit
            
        startf = time.time()
        
        endf = time.time()
        self.timefsplt[0] = endf-startf
        
        #reshaping y
        y = y.reshape(-1,1)

        #Finds maximum y-value for the intial points
        ybst = max(y).reshape(-1,1)
        
        #makes a list of best x-values in chronological order
        x = x.reshape(-1,self.dim)
        xbst = x[np.where(y == ybst)[0],:]
        
        #fitting the gaussian process
        modelsplt = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,n_restarts_optimizer = 10,normalize_y = True)
        
        #fitting model to data
        modelsplt.fit(x, y)

        #creating first matrix of recorded x-values for each split
        xnxt = np.ones((num_max, self.dim))

        #Creates acquisiiton model object
        LCB = LCB_AF(modelsplt, self.dim, self.exp_w, self.descale).LCB

        #TODO: insures minimum value is 1
        init_pts = max(1, int(round(128/self.split, 0)))

        #intial points
        x0 = np.random.uniform(0, sf, (init_pts, self.dim))
        
        #results = Parallel(n_jobs = afcores)(delayed(LCB)(row) for row in self.grid_vertices)
        results = Parallel(n_jobs = afcores)(delayed(LCB)(self.grid_vertices))

        vertices = self.grid_vertices
        for i in range(num_max):
        
            #finds minimum for x-values
            max_index = np.argmax(results)
            xnxt[i] = vertices[max_index]
            bool_matrix = (results != results[max_index])
            results = results[bool_matrix]
            vertices = vertices[bool_matrix]
        
        end = time.time()
        self.timesplt[0] = end-start

        #Total trials to be run
        for i in range(self.trialsplt):
            
            startf = time.time()
            
            #Calculates the y-value for the x-values (one in each grid)
            ynxt = Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(x)) for x in xnxt)
            
            endf = time.time()
            self.timefsplt[i+1] = self.timefsplt[i]+(endf-startf)
            
            #next y-values
            ynxt = ynxt.reshape(-1, 1)
            
            #adding new data points to data set
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt])
            
            #TODO: need to change the ynxt
            ybst = np.vstack([ybst, max(ynxt).reshape(-1, 1)])
            
            row = xnxt[np.where(ynxt == max(ynxt))[0]]
            xbst = np.vstack([xbst, row])

            #fitting guassian process to new data
            modelsplt.fit(x, y)
            
            x0 = np.random.uniform(0, sf, (init_pts, self.dim))

            #for each split, finds the max for the acquisiton function
            results = Parallel(n_jobs = afcores)(delayed(LCB)(self.grid_vertices))

            for j in range(num_max):
            
                max_index = np.argmax(results)
                xnxt[j] = vertices[max_index]
                bool_matrix = (results != results[max_index])
                results = results[bool_matrix]
                vertices = vertices[bool_matrix]

            end = time.time()
            
            self.timesplt[i+1] = end-start

        self.splt_optim = True
        self.modelsplt = modelsplt
        self.xsplt = self.descale(x)
        self.ysplt = y
        self.yspltbst = ybst
        self.xspltbst = self.descale(xbst)


    ####LS-BO with reference model if you don't want one, put in function of zeros 
    # (Level Set)
    def optimizerspltref(self, trials, partition_number, repartition_intervals,
                         x_samps, fcores, afcores, scaling_factor = 1,
                         partition_constraints = None, xinit = None):
        """
        > partition_number is the number of desired partitions
        
        > repartition_intervals is the iteration at which repartitioning is done,
          if repartitioning is not wanted, enter empty list '[]'
        
        > fcores and afcores are the cores used for sampling f (objective function) and optimizing the AF
        
        > x_samps contains the points at which samples of hat{f} are collected
          during repartitioning in order to determine the median, if repartitioning
          is not desired, enter empty array, 'np.array([])', or list, '[]'
          
        > scaling factor sets the size of the 'box' that the input space is scaled
          down to; by default set to scale to unit interval [0, 1]
        
        > partition_constraints should be a numerically indexed dictionary with
          each entry containing a list of the constraints (linear or nonlinear)
          required to set up the desired space partition
         
        > xinit is an array containing the intial points at which to sample
        """
        print('Partitioned Domain with Reference Model BO Run...')
        start = time.time()
        self.trialspltref = trials
        splitref = partition_number
        self.reparts = repartition_intervals
        self.x_s = x_samps
        sf = scaling_factor
        self.consref = partition_constraints
        self.timespltref = np.ones(self.trialspltref+1)
        self.timefspltref = np.ones(self.trialspltref+1)
        x = np.array([]).reshape(0, self.dim)
        refmod = self.refmod['refmod']
        switch = True

        #Creates Gaussian Process Regression model
        modelsplteps = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,
                                                 n_restarts_optimizer = 10,
                                                 normalize_y = True)
        
        #predicting the values
        cons_fun = lambda x: refmod(self.descale(x))+modelsplteps.predict(x.reshape(1, 2))
        
        ###
        # Accessing elements of self.consref
        # Then measures if lower bound is less than function which should be less than upper bounds
        # Else, add 10^6 to the res

        def intpts(x, i):
            l = str(i+1)
            res = 0
            for j in self.consref[l]:
                if j.lb < j.fun(x) < j.ub:
                    res+=0
                else:
                    res+=1e6
            return res
            
        if self.consref is None:
            self.consref = {}
            y_s = refmod(self.x_s)
            delta = np.linspace(np.min(y_s), np.max(y_s), splitref+1)
            self.consref['1'] =   [NonlinearConstraint(cons_fun, -np.inf, delta[1])]
            for i in range(1, splitref):
                self.consref[str(i+1)] = [NonlinearConstraint(cons_fun, delta[i], delta[i+1])]
        for i in range(splitref):
            n = 0
            #intial points
            if xinit is not None and switch == True:
                for j in range(len(self.consref[str(i+1)])):    
                    if self.consref[str(i+1)][j].fun(xinit) < self.consref[str(i+1)][j].ub\
                        and self.consref[str(i+1)][j].fun(xinit) > self.consref[str(i+1)][j].lb:
                            n+=1
                if n == len(self.consref[str(i+1)]):
                    x0 = xinit.reshape(1, self.dim)
                    switch = False
            if n != len(self.consref[str(i+1)]):
                x0 = np.random.uniform(0, sf, (10, self.dim))
                # minimizes Acquisition function
                # instead of minimizing; you need to do a grid search, evaluate LCB at every point in design point and find minimum in the array
                # TODO
                # The arguments could have the grid
                # Have LCB(arrays as argument) TODO check this
                # Parallel allows parallel computation
                opt = Parallel(n_jobs = 1)(delayed(minimize)(intpts, x0 = x, args = (i,),method = 'SLSQP',bounds = self.bounds, tol = 1e-6,constraints = self.consref[str(i+1)]) for x in x0)
                x0s = np.array([res.x for res in opt], dtype = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                x0 = x0s[np.argmin(funs)]
            x = np.vstack([x, x0.reshape(1, self.dim)])
        splt = int(x.shape[0]/fcores)
        xbs = np.array(np.ones(fcores), dtype = tuple)
        if fcores == 1:
            xbs[0] = x
        else:
            for i in range(fcores-1):
                xbs[i] = x[i*splt:(i+1)*splt, :]
            xbs[-1] = x[(i+1)*splt:, :]
        startf = time.time()
        y = Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point)) for start_point in xbs)
        if str(type(refmod))=="<class '__main__.Network'>":
            yref = Parallel(n_jobs = fcores)(delayed(refmod)(start_point) for start_point in torch.from_numpy(x).float())
            endf = time.time()
            yref = torch.hstack(yref[:]).T.reshape(-1, 1).data.numpy()
        else:
            yref = Parallel(n_jobs = fcores)(delayed(refmod)(self.descale(start_point)) for start_point in xbs)
            endf = time.time()
            yref = np.hstack(yref[:]).T.reshape(-1,1)
        y = np.hstack(y[:]).T.reshape(-1,1)
        eps = y-yref
        ybst = min(y).reshape(-1,1)
        self.timefspltref[0] = endf-startf
        modelsplteps.fit(x, eps)
        xnxt = np.ones((splitref, self.dim))
        LCB = LCB_AF(modelsplteps, self.dim, self.exp_w, self.descale, **self.refmod).LCB
        init_pts = max(1, int(round(128/splitref, 0)))
        x0 = np.random.uniform(0, sf, (init_pts, self.dim))
        for i in range(splitref):
            x0 = np.random.uniform(-sf, sf, (100, self.dim))
            opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB, x0 = start_points,
                                                          method = 'SLSQP',
                                                          bounds = self.bounds, tol = 1e-6,
                                                          constraints = self.consref[str(i+1)])
                                        for start_points in x0)
            xnxts = np.array([res.x for res in opt], dtype = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            sts = np.array([res.success for res in opt])
            funs[np.where(sts==False)] = max(1e6, np.max(funs))
            xnxt[i] = xnxts[np.argmin(funs)]
        xnxtbs = np.array(np.ones(fcores), dtype = tuple)
        end = time.time()
        self.timespltref[0] = end-start
        J = 0
        for i in range(self.trialspltref):
            if fcores == 1:
                xnxtbs[0] = xnxt
            else:
                for j in range(fcores-1):
                    xnxtbs[j] = xnxt[j*splt:(j+1)*splt, :]
                xnxtbs[-1] = xnxt[(j+1)*splt:, :]
            startf = time.time()
            ynxt =  Parallel(n_jobs = fcores)(delayed(self.system)(self.descale(start_point)) for start_point in xnxtbs)
            if str(type(refmod)) == "<class '__main__.Network'>":
                yref = Parallel(n_jobs = fcores)(delayed(refmod)(start_point) for start_point in torch.from_numpy(xnxt).float())
                endf = time.time()
                yref = torch.hstack(yref[:]).T.reshape(-1, 1).data.numpy()
            else:
                yref =  Parallel(n_jobs = fcores)(delayed(refmod)(self.descale(start_point)) for start_point in xnxtbs)
                endf = time.time()
                yref = np.hstack(yref[:]).T.reshape(-1, 1)
            ynxt = np.hstack(ynxt[:]).T.reshape(-1,1)
            epsnxt = ynxt-yref
            self.timefspltref[i+1] = self.timefspltref[i]+(endf-startf)
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt])
            eps = np.vstack([eps, epsnxt])
            ybst = np.vstack([ybst, min(ynxt).reshape(-1, 1)])
            modelsplteps.fit(x, eps)
            if i+2 in self.reparts:
                J+=1
                y_s = refmod(self.x_s)+modelsplteps.predict(self.scale(self.x_s))
                med = np.median(y_s)
                for j in range(J):
                    idx = np.where(y_s <= med)
                    y_s = y_s[idx]
                    med = np.median(y_s)
                delta = np.linspace(np.min(y_s), np.max(y_s), splitref+1)
                self.consref['1'] = [NonlinearConstraint(cons_fun, -np.inf, delta[1])]
                for j in range(1, splitref):
                    self.consref[str(j+1)] = [NonlinearConstraint(cons_fun, delta[j], delta[j+1])]
            x0 = np.random.uniform(0, sf, (init_pts, self.dim))
            for j in range(splitref):
                opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB, x0 = start_points,
                                                              method = 'SLSQP',
                                                              bounds = self.bounds, tol = 1e-6,
                                                              constraints = self.consref[str(j+1)])
                                            for start_points in x0)
                xnxts = np.array([res.x for res in opt], dtype = 'float')
                funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                sts = np.array([res.success for res in opt])
                funs[np.where(sts==False)] = max(1e6, np.max(funs))
                xnxt[j] = xnxts[np.argmin(funs)]
            end = time.time()
            self.timespltref[i+1] = end-start
        self.splt_optimref = True
        self.modelsplteps = modelsplteps
        self.xspltref = self.descale(x)
        self.yspltref = y
        self.yspltbstref = ybst
        self.epssplt = eps
    

    ### 
    #VP-BO
    #
    #
    def optimizerspltvar(self, trials, split_num, liminit, scaling_factor, fcores, afcores, xinit  = None):
        """
        VP-BO approach developed by us and detailed in González, L.D. et al. The partitions are
        made using the 'split_num' argument; this sets the number of blocks that the variables
        are split into. The 'liminit' argument sets the initial value for the x_{-k} variables.
        Note that when introducing the design variables, x, they should be ordered by subsystem.
        For example if d = 4, and split_num = 2, introducing x = (x_a, x_b, x_c, x_d) will set
        x_1 = (x_a, x_b) and x_{-1}] = (x_c, x_d), and x_2 = (x_c, x_d) and x_{-2} = (x_a, x_b).
        The 'fcores' argument sets the number of cores used for parallel experimentation, and
        the 'afcores' argument sets the cores used for optimizing the AF
        """
        print('Partitioned Variables BO Run...')

        start = time.time()
        
        #number of trials
        self.trialspltvar = trials
        #number of split variables
        self.splitvar = split_num

        self.timespltvar = np.zeros(self.trialspltvar+1)
        self.timefspltvar = np.zeros(self.trialspltvar+1)
        
        #equals dimensions divided by numbers of splitting
        div = int(self.dim/self.splitvar)
        print('div ' + str(div))
        sf = scaling_factor

        refmod = self.dist_ref['distrefmod']

        #sets intial value for the variables
        x = liminit * np.ones((self.splitvar, self.dim))
        print('x intial')
        print(x)

        #starts locking the variable
        lwr = x.copy()
        #essentially equal
        upr = x.copy()+1e-6

        #initial values
        print('splitvar')
        print(self.splitvar)
        
        # creates the number of random points per partition
        # unlocks the variables
        for i in range(self.splitvar):
            if xinit is None:
                #random distribution of sample points
                x[i, i*div:(i+1)*div] = np.random.uniform(0, 1, (1, div))
            else:
                # array of self.dim length
                xinit = xinit.reshape(1, self.dim)
                x[i, i*div:(i+1)*div] = xinit[0, i*div:(i+1)*div]

            # lower bound is 0 for each partition
            lwr[i, i*div:(i+1)*div] = 0
            
            # upper bound is 1 for each partition
            upr[i, i*div:(i+1)*div] = sf

        print('x')
        print(x)
        #adds the intial x back to the randomly taken from above
        x = np.vstack([x, liminit])

        print(x)

        splt = int(x.shape[0]/fcores)

        xbs = np.array(np.ones(fcores), dtype = tuple)


        if fcores == 1:
            xbs[0] = x
        else:
            for i in range(fcores-1):
                xbs[i] = x[i*splt:(i+1)*splt, :]
            xbs[-1] = x[(i+1)*splt:, :]

        startf = time.time()

        #Y-value
        print(xbs)


        y = Parallel(n_jobs = fcores)(delayed(self.distmod)(self.descale(start_point)) for start_point in xbs)

        print(y)
        #reference model
        if str(type(refmod))=="<class '__main__.Network'>":
            yref = Parallel(n_jobs = fcores)(delayed(refmod)(start_point) for start_point in torch.from_numpy(x).float())
            endf = time.time()
            yref = torch.hstack(yref[:]).T.reshape(-1, 1).data.numpy()
        else:
            yref = Parallel(n_jobs = fcores)(delayed(refmod)(self.descale(start_point)) for start_point in xbs)
            endf = time.time()
            yref = np.vstack(yref[:])
        self.timefspltvar[0] = endf-startf

        y = np.vstack(y[:])
        print("y")
        print(y)
        print("yref")
        print(yref)
        #eps is initialized
        eps = y-yref

        ybst = np.min(y, axis = 0).reshape(-1, 1).T
        modelspltvar = {}
        bndsvar = {}
        LCB = {}
        xnxt = x.copy()
        init_pts = int(round(128**(div/self.dim)))
        x0 = np.random.uniform(0, sf, (init_pts, self.dim))

        #loops for how many splits in domain of variables
        for i in range(self.splitvar):
            #Guassian Process object stored in a array of Guassian Processes
            modelspltvar[str(i+1)] = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,n_restarts_optimizer = 10,normalize_y = True)

            # fitted
            print('x')
            print(x)
            print('eps')
            print(eps)
            print('index: '+ str(i))
            print(eps[:,0])
            #TODO: eps[:,i]
            modelspltvar[str(i+1)].fit(x, eps[:,i])

            # Variables are bounded depending on the partition
            bndsvar[str(i+1)] = Bounds(lwr[i], upr[i])

            # Creates the LCB object and put into array
            LCB[str(i+1)] = LCB_AF(modelspltvar[str(i+1)], self.dim, self.exp_w,self.descale, **{'refmod': self.dist_ref['distrefmod'+str(i+1)]}).LCB
            
            #Calculates the x-values that seem to be the best
            opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB[str(i+1)], x0 = start_point,method = 'L-BFGS-B',bounds = bndsvar[str(i+1)])for start_point in x0)
            #TODO: constraints for create_axis
            #self.create_axis(lwr[i],upr[i],index = i)
            #self.iterate(index= i)

            uncert = LCB[str(i+1)](self.grid_vertices[i])
            xnxt[i] = self.grid_vertices[i][np.where(uncert == np.min(uncert))]
            
            #creates an array
            #xnxts = np.array([res.x for res in opt], dtype = 'float')

            #finds the y-value for AF
            #funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])

            #finds the best x - value
            #xnxt[i] = xnxts[np.argmin(funs)]

            #I got no idea
            #xnxt[-1, i*div:(i+1)*div] = xnxts[np.argmin(funs), i*div:(i+1)*div]
        

        xnxtbs = np.array(np.ones(fcores), dtype = tuple)
        
        end = time.time()
        self.timespltvar[0] = end-start

        # 
        for i in range(self.trialspltvar):
            if fcores == 1:
                xnxtbs[0] = xnxt
            else:
                for j in range(fcores-1):
                    xnxtbs[j] = xnxt[j*splt:(j+1)*splt, :]
                xnxtbs[-1] = xnxt[(j+1)*splt:, :]
            startf = time.time()

            #Calculates the y-values
            ynxt = Parallel(n_jobs = fcores)(delayed(self.distmod)(self.descale(start_point)) for start_point in xnxtbs)

            #using model to find y-reference models
            if str(type(refmod))=="<class '__main__.Network'>":
                yref = Parallel(n_jobs = fcores)(delayed(refmod)(start_point) for start_point in torch.from_numpy(x).float())
                endf = time.time()
                yref = torch.hstack(yref[:]).T.reshape(-1, 1).data.numpy()
            else:
                yref = Parallel(n_jobs = fcores)(delayed(refmod)(self.descale(start_point)) for start_point in xnxtbs)
                endf = time.time()
                yref = np.vstack(yref[:])

            #receive next y-value fr each domain
            ynxt = np.vstack(ynxt[:])

            # subtracting the actual y-values from the model y-values
            epsnxt = ynxt-yref

            self.timefspltvar[i+1] = self.timefspltvar[i]+(endf-startf)

            #I dont know what is going on
            for j in range(self.splitvar):
                if any(ynxt[:, j] < min(y[:, j])):
                    lwr[j] = xnxt[np.argmin(ynxt[:, j])]
                    lwr[j, j*div:(j+1)*div] = 0
                    upr[j] = xnxt[np.argmin(ynxt[:, j])]+1e-6
                    upr[j, j*div:(j+1)*div] = sf
            
            #adding the next x & y-values & y-value - y-ref values
            x = np.vstack([x, xnxt])
            y = np.vstack([y, ynxt])
            eps = np.vstack([eps, epsnxt])


            ybst = np.vstack([ybst, np.min(ynxt, axis = 0).reshape(-1,1).T])

            #random points  
            x0 = np.random.uniform(0, sf, (init_pts, self.dim))

            # loops through each splitvar     
            for j in range(self.splitvar):
                # fitting x domain to the difference bewteen the reference model and the y-values
                modelspltvar[str(j+1)].fit(x, eps[:, j])
                #bounds 
                bndsvar[str(j+1)] = Bounds(lwr[j], upr[j])
                #He bounds each with Parallel:
                
                #opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB[str(j+1)], x0 = start_point,method = 'L-BFGS-B',bounds = bndsvar[str(j+1)]) for start_point in x0)
                
                self.create_axis(lwr[j],upr[j],index = j)
                self.iterate(index=j)

                uncert = LCB[self.grid_vertices[i]]
                xnxt = self.grid_vertices[i][np.where(uncert == np.min(uncert))]
                #LCB(self.grid_vertices[j])
                #xnxts = np.array([res.x for res in opt], dtype = 'float')
                #funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                #xnxt[j] = xnxts[np.argmin(funs)]
                #xnxt[-1, j*div:(j+1)*div] = xnxts[np.argmin(funs), j*div:(j+1)*div]

            end = time.time()
            self.timespltvar[i+1] = end-start

        self.spltvar_optim = True
        self.modelspltvar = modelspltvar
        self.xspltvar = self.descale(x)
        self.yspltvar = y
        self.yspltvarbst = ybst
        
    def optimizerembd(self, trials, var_num, include_x, fun, scaling_factor, fcores, afcores, xinit = None):
        """
        Work in Progress
        """
        print('Embedded function BO Run...')
        start = time.time()
        self.trialsembd = trials
        self.var_num = var_num
        self.include_x = include_x
        self.fun = fun
        sf = scaling_factor
        self.timembd = np.zeros(self.trialsembd+1)
        self.timefembd = np.zeros(self.trialsembd+1)
        if xinit is None:
            x = np.random.uniform(0, sf, (1, self.dim))
        else:
            x = xinit.reshape(1, self.dim)
        startf = time.time()
        d = self.system(self.descale(x))
        endf = time.time()
        self.timefembd[0] = endf-startf
        modembd = {}
        for i in range(self.var_num):
            modembd[str(i+1)] = gpr.GaussianProcessRegressor(self.kernel, alpha = 1e-6,
                                                             n_restarts_optimizer = 10,
                                                             normalize_y = True)
            modembd[str(i+1)].fit(x, d[:, i].reshape(-1, 1))
        LCB = LCB_EMBD(modembd, self.var_num, self.dim, self.exp_w, self.fun, self.descale, self.include_x).LCB
        x0 = np.random.uniform(0, sf, (100, self.dim))
        opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB, x0 = start_point,
                                                           method = 'L-BFGS-B',
                                                           bounds = self.bounds)
                                         for start_point in x0)
        xnxts = np.array([res.x for res in opt], dtype = 'float')
        funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
        xnxt = xnxts[np.argmin(funs)].reshape(1, self.dim)
        end = time.time()
        self.timembd[0] = end-start
        for i in range(self.trialsembd):
            startf = time.time()
            dnxt = self.system(self.descale(xnxt))
            endf = time.time()
            self.timefembd[i+1] = self.timefembd[i]+(endf-startf)
            x = np.vstack([x, xnxt])
            d = np.vstack([d, dnxt])
            for j in range(self.var_num):
                modembd[str(j+1)].fit(x, d[:, j])
            x0 = np.random.uniform(0, sf, (100, self.dim))
            opt = Parallel(n_jobs = afcores)(delayed(minimize)(LCB, x0 = start_point,
                                                               method = 'L-BFGS-B',
                                                               bounds = self.bounds)
                                             for start_point in x0)
            xnxts = np.array([res.x for res in opt], dtype = 'float')
            funs = np.array([np.atleast_1d(res.fun)[0] for res in opt])
            xnxt = xnxts[np.argmin(funs)].reshape(1, self.dim)
            end = time.time()
            self.timembd[i+1] = end-start
        self.embd_optim = True
        self.xembd = self.descale(x)
        self.yembd = d
        if self.include_x:
            self.yembd = np.hstack([self.yembd, self.xembd])
        self.fembd = self.fun(self.yembd)
        self.modembd = modembd
    
    def plots(self, figure_name):
        itr = np.arange(1, self.trialsgp+2, 1)
        yliml = min(self.ygp)-0.01*abs(min(self.ygp))
        ylimu = max(self.ygp)+0.01*abs(max(self.ygp))
        fig, ax1 = pyp.subplots(1, 1, figsize=(11, 8.5))
        ax1.grid(color='gray', axis='both', alpha = 0.25)
        ax1.set_axisbelow(True)
        ax1.set_xlim((1, self.trialsgp+1))
        ax1.set_xlabel('Sample Number', fontsize = 24)
        pyp.xticks(fontsize = 24)
        ax1.set_ylabel('Operating cost (MM USD/yr)', fontsize = 24)
        pyp.yticks(fontsize = 24);
        ax1.scatter(itr, self.ygp, marker = 'o', color = 'white', edgecolor = 'black',
                    zorder = 3, s = 200);
        ax1.plot(itr, self.ygp, color = 'black', linewidth = 3, label = 'BO');
        if self.ref_optim:
            itr = np.arange(1, self.trialsref+2, 1)
            ax1.scatter(itr, self.ytru, marker = 'o', color = 'white', edgecolor = 'blue',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.ytru, color = 'blue', linewidth = 3, label = 'Ref-BO');
            yliml = min(yliml, min(self.ytru)-0.01*abs(min(self.ytru)))
            ylimu = max(ylimu, max(self.ytru)+0.01*abs(max(self.ytru)))
        if self.splt_optim:
            itr = np.arange(1, self.trialsplt+2, 1)
            ax1.scatter(itr, self.yspltbst, marker = 'o', color = 'white', edgecolor = 'purple',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.yspltbst, color = 'purple', linewidth = 3, label = 'Partioned BO');
            yliml = min(yliml, min(self.yspltbst)-0.01*abs(min(self.yspltbst)))
            ylimu = max(ylimu, max(self.yspltbst)+0.01*abs(max(self.yspltbst)))
        if self.splt_optimref:
            itr = np.arange(1, self.trialspltref+2, 1)
            ax1.scatter(itr, self.yspltbstref, marker = 'o', color = 'white', edgecolor = 'gray',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.yspltbstref, color = 'gray', linewidth = 3, label = 'LS-BO');
            yliml = min(yliml, min(self.yspltbstref)-0.01*abs(min(self.yspltbstref)))
            ylimu = max(ylimu, max(self.yspltbstref)+0.01*abs(max(self.yspltbstref)))
        if self.spltvar_optim:
            itr = np.arange(1, self.trialspltvar+2, 1)
            ax1.scatter(itr, self.yspltvarbst[:, -1], marker = 'o', color = 'white', edgecolor = 'brown',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.yspltvarbst[:, -1], color = 'brown', linewidth = 3, label = 'VP-BO');
            yliml = min(yliml, min(self.yspltvarbst[:, -1])-0.01*abs(min(self.yspltvarbst[:, -1])))
            ylimu = max(ylimu, max(self.yspltvarbst[:, -1])+0.01*abs(max(self.yspltvarbst[:, -1])))
        if self.hyp_optim:
            itr = np.arange(1, self.trialshyp+2, 1)
            ax1.scatter(itr, self.yhypbst, marker = 'o', color = 'white', edgecolor = 'green',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.yhypbst, color = 'green', linewidth = 3, label = 'HS-BO');
            yliml = min(yliml, min(self.yhypbst)-0.01*abs(min(self.yhypbst)))
            ylimu = max(ylimu, max(self.yhypbst)+0.01*abs(max(self.yhypbst)))
        if self.expwl_optim:
            itr = np.arange(1, self.trialsexpw+2, 1)
            ax1.scatter(itr, self.ybstexpw, marker = 'o', color = 'white', edgecolor = 'red',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.ybstexpw, color = 'red', linewidth = 3, label = 'HP-BO');
            yliml = min(yliml, min(self.ybstexpw)-0.01*abs(min(self.ybstexpw)))
            ylimu = max(ylimu, max(self.ybstexpw)+0.01*abs(max(self.ybstexpw)))
        if self.nmc_optim:
            itr = np.arange(1, self.trialsnmc+2, 1)
            ax1.scatter(itr, self.ybstnmc, marker = 'o', color = 'white', edgecolor = 'pink',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.ybstnmc, color = 'pink', linewidth = 3, label = 'MC-BO');
            yliml = min(yliml, min(self.ybstnmc)-0.01*abs(min(self.ybstnmc)))
            ylimu = max(ylimu, max(self.ybstnmc)+0.01*abs(max(self.ybstnmc)))
        if self.qBO_optim:
            itr = np.arange(1, self.trialsqBO+2, 1)
            ax1.scatter(itr, self.ybstqBO, marker = 'o', color = 'white', edgecolor = 'gold',
                    zorder = 3, s = 200);
            ax1.plot(itr, self.ybstqBO, color = 'gold', linewidth = 3, label = 'q-BO');
            yliml = min(yliml, min(self.ybstqBO)-0.01*abs(min(self.ybstqBO)))
            ylimu = max(ylimu, max(self.ybstqBO)+0.01*abs(max(self.ybstqBO)))
        if self.embd_optim:
            itr = np.arange(1, self.trialsembd+2, 1)
            ax1.scatter(itr, self.fembd, marker = 'o', color = 'white', edgecolor = 'lime',
                        zorder = 3, s = 200);
            ax1.plot(itr, self.fembd, color = 'lime', linewidth = 3, label = 'Embedded f BO')
            yliml = min(yliml, min(self.fembd)-0.01*abs(min(self.fembd)))
            ylimu = max(ylimu, max(self.fembd)+0.01*abs(max(self.fembd)))
        
        ax1.set_ylim(yliml, ylimu)
        pyp.legend(loc = 'upper right')
        pyp.show()
        pyp.savefig(figure_name+'.png', dpi = 300, edgecolor = 'white',
                    bbox_inches = 'tight', pad_inches = 0.1);
        
    def plotstime(self, figure_name):
        yliml = min(self.ygp)-0.01*abs(min(self.ygp))
        ylimu = max(self.ygp)+0.01*abs(max(self.ygp))
        xlimu = round(self.timegp[-1]+1, 0)
        fig, ax1 = pyp.subplots(1, 1, figsize=(11, 8.5))
        ax1.grid(color='gray', axis='both', alpha = 0.25)
        ax1.set_axisbelow(True)
        ax1.set_xlabel('Time (s)', fontsize = 24)
        pyp.xticks(fontsize = 24)
        ax1.set_ylabel('Operating cost (MM USD/hr)', fontsize = 24)
        pyp.yticks(fontsize = 24);
        ax1.scatter(self.timegp, self.ygp, marker = 'o', color = 'white', edgecolor = 'black',
                    zorder = 3, s = 200);
        ax1.plot(self.timegp, self.ygp, color = 'black', linewidth = 3, label = 'BO');
        if self.ref_optim:
            ax1.scatter(self.timeref, self.ytru, marker = 'o', color = 'white', edgecolor = 'blue',
                    zorder = 3, s = 200);
            ax1.plot(self.timeref, self.ytru, color = 'blue', linewidth = 3, label = 'Ref-BO');
            yliml = min(yliml, min(self.ytru)-0.01*abs(min(self.ytru)))
            ylimu = max(ylimu, max(self.ytru)+0.01*abs(max(self.ytru)))
            xlimu = max(xlimu, self.timeref[-1]+1)
        if self.splt_optim:
            ax1.scatter(self.timesplt, self.yspltbst, marker = 'o', color = 'white', edgecolor = 'purple',
                    zorder = 3, s = 200);
            ax1.plot(self.timesplt, self.yspltbst, color = 'purple', linewidth = 3, label = 'Partioned BO');
            yliml = min(yliml, min(self.yspltbst)-0.01*abs(min(self.yspltbst)))
            ylimu = max(ylimu, max(self.yspltbst)+0.01*abs(max(self.yspltbst)))
            xlimu = max(xlimu, self.timesplt[-1]+1)
        if self.splt_optimref:
            ax1.scatter(self.timespltref, self.yspltbstref, marker = 'o', color = 'white', edgecolor = 'gray',
                    zorder = 3, s = 200);
            ax1.plot(self.timespltref, self.yspltbstref, color = 'gray', linewidth = 3, label = 'LS-BO');
            yliml = min(yliml, min(self.yspltbstref)-0.01*abs(min(self.yspltbstref)))
            ylimu = max(ylimu, max(self.yspltbstref)+0.01*abs(max(self.yspltbstref)))
            xlimu = max(xlimu, self.timespltref[-1]+1)
        if self.spltvar_optim:
            ax1.scatter(self.timespltvar, self.yspltvarbst[:, -1], marker = 'o', color = 'white', edgecolor = 'brown',
                    zorder = 3, s = 200);
            ax1.plot(self.timespltvar, self.yspltvarbst[:, -1], color = 'brown', linewidth = 3, label = 'VP-BO');
            yliml = min(yliml, min(self.yspltvarbst[:, -1])-0.01*abs(min(self.yspltvarbst[:, -1])))
            ylimu = max(ylimu, max(self.yspltvarbst[:, -1])+0.01*abs(max(self.yspltvarbst[:, -1])))
            xlimu = max(xlimu, self.timespltvar[-1]+1)
        if self.hyp_optim:
            ax1.scatter(self.timehyp, self.yhypbst, marker = 'o', color = 'white', edgecolor = 'green',
                    zorder = 3, s = 200);
            ax1.plot(self.timehyp, self.yhypbst, color = 'green', linewidth = 3, label = 'HS-BO');
            yliml = min(yliml, min(self.yhypbst)-0.01*abs(min(self.yhypbst)))
            ylimu = max(ylimu, max(self.yhypbst)+0.01*abs(max(self.yhypbst)))
            xlimu = max(xlimu, self.timehyp[-1]+1)
        if self.expwl_optim:
            ax1.scatter(self.timexpw, self.ybstexpw, marker = 'o', color = 'white', edgecolor = 'red',
                    zorder = 3, s = 200);
            ax1.plot(self.timexpw, self.ybstexpw, color = 'red', linewidth = 3, label = 'HP-BO');
            yliml = min(yliml, min(self.ybstexpw)-0.01*abs(min(self.ybstexpw)))
            ylimu = max(ylimu, max(self.ybstexpw)+0.01*abs(max(self.ybstexpw)))
            xlimu = max(xlimu, self.timexpw[-1]+1)
        if self.nmc_optim:
            ax1.scatter(self.timenmc, self.ybstnmc, marker = 'o', color = 'white', edgecolor = 'pink',
                    zorder = 3, s = 200);
            ax1.plot(self.timenmc, self.ybstnmc, color = 'pink', linewidth = 3, label = 'MC-BO');
            yliml = min(yliml, min(self.ybstnmc)-0.01*abs(min(self.ybstnmc)))
            ylimu = max(ylimu, max(self.ybstnmc)+0.01*abs(max(self.ybstnmc)))
            xlimu = max(xlimu, self.timenmc[-1]+1)
        if self.qBO_optim:
            ax1.scatter(self.timeqBO, self.ybstqBO, marker = 'o', color = 'white', edgecolor = 'gold',
                    zorder = 3, s = 200);
            ax1.plot(self.timeqBO, self.ybstqBO, color = 'gold', linewidth = 3, label = 'q-BO');
            yliml = min(yliml, min(self.ybstqBO)-0.01*abs(min(self.ybstqBO)))
            ylimu = max(ylimu, max(self.ybstqBO)+0.01*abs(max(self.ybstqBO)))
            xlimu = max(xlimu, self.timeqBO[-1]+1)
        if self.embd_optim:
            ax1.scatter(self.timembd, self.fembd, marker = 'o', color = 'white', edgecolor = 'lime',
                        zorder = 3, s = 200);
            ax1.plot(self.timembd, self.fembd, color = 'lime', linewidth = 3, label = 'Embedded f BO')
            yliml = min(yliml, min(self.fembd)-0.01*abs(min(self.fembd)))
            ylimu = max(ylimu, max(self.fembd)+0.01*abs(max(self.fembd)))
            xlimu = max(xlimu, self.timembd[-1]+1)
                
        ax1.set_ylim(yliml, ylimu)
        ax1.set_xlim(0, xlimu)
        pyp.legend(loc = 'upper right')
        pyp.show()
        pyp.savefig(figure_name+'_time.png', dpi = 300, edgecolor = 'white',
                    bbox_inches = 'tight', pad_inches = 0.1);
        
    def plotexptime(self, figure_name):
        yliml = min(self.ygp)-0.01*abs(min(self.ygp))
        ylimu = max(self.ygp)+0.01*abs(max(self.ygp))
        xlimu = round(self.timefgp[-1]+1, 0)
        fig, ax1 = pyp.subplots(1, 1, figsize=(11, 8.5))
        ax1.grid(color='gray', axis='both', alpha = 0.25)
        ax1.set_axisbelow(True)
        ax1.set_xlabel('Time (s)', fontsize = 24)
        pyp.xticks(fontsize = 24)
        ax1.set_ylabel('Operating cost (MM USD/hr)', fontsize = 24)
        pyp.yticks(fontsize = 24);
        ax1.scatter(self.timefgp, self.ygp, marker = 'o', color = 'white', edgecolor = 'black',
                    zorder = 3, s = 200);
        ax1.plot(self.timefgp, self.ygp, color = 'black', linewidth = 3, label = 'BO');
        if self.ref_optim:
            ax1.scatter(self.timefref, self.ytru, marker = 'o', color = 'white', edgecolor = 'blue',
                    zorder = 3, s = 200);
            ax1.plot(self.timefref, self.ytru, color = 'blue', linewidth = 3, label = 'Ref-BO');
            yliml = min(yliml, min(self.ytru)-0.01*abs(min(self.ytru)))
            ylimu = max(ylimu, max(self.ytru)+0.01*abs(max(self.ytru)))
            xlimu = max(xlimu, self.timefref[-1]+1)
        if self.splt_optim:
            ax1.scatter(self.timefsplt, self.yspltbst, marker = 'o', color = 'white', edgecolor = 'purple',
                    zorder = 3, s = 200);
            ax1.plot(self.timefsplt, self.yspltbst, color = 'purple', linewidth = 3, label = 'Partioned BO');
            yliml = min(yliml, min(self.yspltbst)-0.01*abs(min(self.yspltbst)))
            ylimu = max(ylimu, max(self.yspltbst)+0.01*abs(max(self.yspltbst)))
            xlimu = max(xlimu, self.timefsplt[-1]+1)
        if self.splt_optimref:
            ax1.scatter(self.timefspltref, self.yspltbstref, marker = 'o', color = 'white', edgecolor = 'gray',
                    zorder = 3, s = 200);
            ax1.plot(self.timefspltref, self.yspltbstref, color = 'gray', linewidth = 3, label = 'LS-BO');
            yliml = min(yliml, min(self.yspltbstref)-0.01*abs(min(self.yspltbstref)))
            ylimu = max(ylimu, max(self.yspltbstref)+0.01*abs(max(self.yspltbstref)))
            xlimu = max(xlimu, self.timefspltref[-1]+1)
        if self.spltvar_optim:
            ax1.scatter(self.timefspltvar, self.yspltvarbst[:, -1], marker = 'o', color = 'white', edgecolor = 'brown',
                    zorder = 3, s = 200);
            ax1.plot(self.timefspltvar, self.yspltvarbst[:, -1], color = 'brown', linewidth = 3, label = 'VP-BO');
            yliml = min(yliml, min(self.yspltvarbst[:, -1])-0.01*abs(min(self.yspltvarbst[:, -1])))
            ylimu = max(ylimu, max(self.yspltvarbst[:, -1])+0.01*abs(max(self.yspltvarbst[:, -1])))
            xlimu = max(xlimu, self.timefspltvar[-1]+1)
        if self.hyp_optim:
            ax1.scatter(self.timefhyp, self.yhypbst, marker = 'o', color = 'white', edgecolor = 'green',
                    zorder = 3, s = 200);
            ax1.plot(self.timefhyp, self.yhypbst, color = 'green', linewidth = 3, label = 'HS-BO');
            yliml = min(yliml, min(self.yhypbst)-0.01*abs(min(self.yhypbst)))
            ylimu = max(ylimu, max(self.yhypbst)+0.01*abs(max(self.yhypbst)))
            xlimu = max(xlimu, self.timefhyp[-1]+1)
        if self.expwl_optim:
            ax1.scatter(self.timefexpw, self.ybstexpw, marker = 'o', color = 'white', edgecolor = 'red',
                    zorder = 3, s = 200);
            ax1.plot(self.timefexpw, self.ybstexpw, color = 'red', linewidth = 3, label = 'HP-BO');
            yliml = min(yliml, min(self.ybstexpw)-0.01*abs(min(self.ybstexpw)))
            ylimu = max(ylimu, max(self.ybstexpw)+0.01*abs(max(self.ybstexpw)))
            xlimu = max(xlimu, self.timefexpw[-1]+1)
        if self.nmc_optim:
            ax1.scatter(self.timefnmc, self.ybstnmc, marker = 'o', color = 'white', edgecolor = 'pink',
                    zorder = 3, s = 200);
            ax1.plot(self.timefnmc, self.ybstnmc, color = 'pink', linewidth = 3, label = 'MC-BO');
            yliml = min(yliml, min(self.ybstnmc)-0.01*abs(min(self.ybstnmc)))
            ylimu = max(ylimu, max(self.ybstnmc)+0.01*abs(max(self.ybstnmc)))
            xlimu = max(xlimu, self.timefnmc[-1]+1)
        if self.qBO_optim:
            ax1.scatter(self.timefqBO, self.ybstqBO, marker = 'o', color = 'white', edgecolor = 'gold',
                    zorder = 3, s = 200);
            ax1.plot(self.timefqBO, self.ybstqBO, color = 'gold', linewidth = 3, label = 'q-BO');
            yliml = min(yliml, min(self.ybstqBO)-0.01*abs(min(self.ybstqBO)))
            ylimu = max(ylimu, max(self.ybstqBO)+0.01*abs(max(self.ybstqBO)))
            xlimu = max(xlimu, self.timefqBO[-1]+1)
        if self.embd_optim:
            ax1.scatter(self.timefembd, self.fembd, marker = 'o', color = 'white', edgecolor = 'lime',
                        zorder = 3, s = 200);
            ax1.plot(self.timefembd, self.fembd, color = 'lime', linewidth = 3, label = 'Embedded f BO')
            yliml = min(yliml, min(self.fembd)-0.01*abs(min(self.fembd)))
            ylimu = max(ylimu, max(self.fembd)+0.01*abs(max(self.fembd)))
            xlimu = max(xlimu, self.timefembd[-1]+1)
                
        ax1.set_ylim(yliml, ylimu)
        ax1.set_xlim(0, xlimu)
        pyp.legend(loc = 'upper right')
        pyp.show()
        pyp.savefig(figure_name+'_exptime.png', dpi = 300, edgecolor = 'white',
                    bbox_inches = 'tight', pad_inches = 0.1);