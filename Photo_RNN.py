import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jax.scipy.integrate import trapezoid
from scipy.stats import linregress
from armored.models import *
from armored.preprocessing import *
import seaborn as sns
import time
import random
import myBO
#importing myAlgae1
import os
file_path = os.path.abspath('/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Algae Model')
import sys
sys.path.append(file_path)
import Photo_mechanistic_modeling_biomass as PM
from Grid_search_for_opt import iterate, create_axis, test_algae_BO_2D_Nitrogen
import Photo_high_dim as PM_5
import Photo_4_dim as PM_4
import Photo_3_dim as PM_3
import Photo_2_dim as PM_2

def predict_biomass_v_time():
    #Take from csv file
    df = pd.read_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/BO Algos/11903_mass_data.csv")
    #returns first 4 rows
    df.head(4)

    #String of dependent variable
    Ind_var = np.array(['biomass'])

    d_var = []

    #dependent variables
    controls = np.array(['light intensity'])

    #creating a matrix of all variables
    system_variables = np.concatenate((Ind_var,d_var,controls))

    #finds unique experiments in first column
    treatments = np.unique(df.Experiments.values)

    #select 3 unique experiments
    n_test = 1
    test_set = np.random.choice(treatments,n_test,replace = False)

    #returns indices of the 3 chosen experiments
    test_inds = np.in1d(df.Experiments.values, test_set)
    #Creates dataframe for non-tested experiments
    train_df = df.iloc[~test_inds].copy()
    #Creates dataframe for test experiments
    test_df = df.iloc[test_inds].copy()

    #Creating manual dataframe with unique starting condition
    '''
    data = [[len(treatments)+1,0,1000,.0555]]
    test_df = pd.DataFrame(data,columns = ['Experiments','Time','light intensity','biomass'])
    '''

    train_data,exp_names,N = format_data(train_df,Ind_var,d_var,controls)
    test_data, _, _ = format_data(test_df,Ind_var,d_var,controls)

    scaler = ZeroMaxScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    #settubg up rnn object
    rnn = RNN.miRNN(n_species = len(Ind_var),n_metabolites = len(d_var), n_controls = len(controls), n_hidden = 30, N=N)
    #Fitting data to rnn
    rnn.fit(train_data)

    #making predicitions
    preds, stdvs, COV = rnn.predict(test_data)

    #reversing scaler transformation
    preds = scaler.inverse_transform(preds)
    stdvs = scaler.inverse_transform_stdv(stdvs)


    test_data = scaler.inverse_transform(test_data)

    params = {'legend.fontsize': 18,
            'figure.figsize': (16,12),
            'lines.linewidth': 4,
            'axes.labelsize': 24,
            'axes.titlesize':24,
            'axes.linewidth': 5,
            'xtick.labelsize': 20,
            'ytick.labelsize':20}

    plt.rcParams.update(params)
    #plt.style.use('seaborn-colorblind')
    plt.rcParams['pdf.fonttype'] = 42

    exp_ind = np.random.choice(np.arange(test_data.shape[0]))
    t_span = [0,12,24,36,48,60,72,83,96,108,120]

    for i, variable in enumerate(d_var):
        var_ind = np.in1d(system_variables, variable)

    print(test_data)
    print(preds.ravel())

    plt.scatter(t_span, test_data[0,:,0].ravel(), color='C{}'.format(1), s=100)
    plt.plot(t_span, preds.ravel(), label="biomass", color='C{}'.format(1))
    plt.fill_between(t_span, preds.ravel() - stdvs.ravel(), 
                        preds.ravel() + stdvs.ravel(), 
                        color='C{}'.format(1), alpha=.3)
    plt.legend(fontsize=28)
    plt.xticks(t_span, fontsize=32)
    plt.xlabel("Time (hr)", fontsize=32)
    plt.yticks(fontsize=32)
    plt.ylabel("Biomass Concentration (g/L)", fontsize=32)
    plt.show()

def BO_RNN():
    '''
    Uses exploration + exploitation in Bayesian Optimization to find the optimized light intensity for Biomass growth
    '''
    params = {'legend.fontsize': 18,
          'figure.figsize': (8, 6),
         'axes.labelsize': 24,
         'axes.titlesize':24,
         'axes.linewidth':5,
         'xtick.labelsize':20,
         'ytick.labelsize':20}
    plt.rcParams.update(params)
    #plt.style.use('seaborn-colorblind')
    plt.rcParams['pdf.fonttype'] = 42

    from numpy.random import default_rng
    rng = default_rng(seed = 123)

    # number of trials 
    n_trials = 5

    # number of dtl cycles 
    n_dtl  = 5

    # define number of initial samples to train on
    n_init = 100

    # number of samples for next experiment 
    n_test = 10

    # number of species in model
    n_s = 1

    # number of resources
    n_r = 0

    # define all system variables 
    species = ['Biomass']
    outputs = []
    sys_var = species + outputs

    # define parameters in the objective function
    obj_params = []#['Light Intensity']

    # define subset of controls
    # Trying control time
    controls = ['Light Intensity']
    system_variables = species + outputs + controls

    # define an objective function (product of endpoint volume * endpoint product concentration)
    # where product concentration is the last column of the predicted output
    #objective = lambda pred, vol: pred[-1, -1]*vol[-1] 
    #objective = PM.BO_intializer
    time_interval = np.linspace(0,96,3)

    def objective(y,obj):
        '''
        print(y.shape)
        print(y[:,0])
        print(time_interval.shape)
        print('trap')
        print(trapezoid(y[:,0]))
        print(trapezoid(y[:,0],time_interval))
        '''
        return trapezoid(y[:,0],time_interval)

    # import data 
    main_df = pd.read_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/BO Algos/Data/algae_light_smaller.csv")
    all_exp_names = main_df.Experiments.values
    # determine random sets of initial experiments
    initial_exps = [rng.choice(np.unique(all_exp_names), n_init, replace=False) for _ in range(n_trials)]
    # function to compute sum of squares error 
    def sse(a, b):
        return np.sum((a-b)**2)
    
    # init dataframe that stores DTL information
    dtl_df = pd.DataFrame()
    dtl_df_R = pd.DataFrame()
    dtl_df_sse = pd.DataFrame()
    elapsed_time = []

    for trial in range(n_trials):

        # format data 
        main_data, main_obj_params, unique_exp_names, N_total = format_data(main_df, species, outputs, controls, obj_params=obj_params)

        # keep track of objective 
        objective_found = []

        # choose random set of training samples
        train_df = main_df.iloc[np.in1d(all_exp_names, initial_exps[trial])].copy()
        train_data, train_obj_params, new_experiments, N = format_data(train_df, species, outputs, controls, obj_params=obj_params)

        # remove training samples from main dataset so that they're not selected more than once
        train_inds = np.in1d(unique_exp_names, new_experiments)
        main_data = main_data[~train_inds]
        main_obj_params  = main_obj_params[~train_inds]
        unique_exp_names = unique_exp_names[~train_inds]

        # compute objectives
        target_found = []
        for sample, train_obj_param in zip(train_data, train_obj_params):
            target_found.append(objective(sample,train_obj_param))
        
        target_found = np.array(target_found)
        print(target_found)
        objective_found.append(np.max(target_found))
        objective_rval = []
        objective_sse  = []

        # Search over full factorial and update model
        for dtl in range(n_dtl):
            print(f"Running trial {trial+1}, cycle {dtl+1}")
        
        # scale train and design space data
        scaler = ZeroMaxScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        main_data_scaled = scaler.transform(main_data)

        # fit model 
        brnn = miRNN(n_species=n_s, n_metabolites=1, n_controls=len(controls), n_hidden=16, N=N)
        brnn.fit(train_data)
        
        # assess prediction performance of end-point product
        pred, stdv, cov = brnn.predict(main_data_scaled)
        pred = scaler.inverse_transform(pred)
        stdv = scaler.inverse_transform(stdv)

        rvalue = linregress(np.array(main_data[:, -1, 0]), pred[:, -1, -1]).rvalue
        sse_value = sse(np.array(main_data[:, -1, 0]), pred[:, -1, -1])
        plt.scatter(np.array(main_data[:, -1, 0]), pred[:, -1, -1], label="R = {:.3f}\nSSE = {:.3f}".format(rvalue, sse_value))
        plt.legend()
        #plt.show()

        objective_rval.append(rvalue)
        objective_sse.append(sse_value)
        
        # randomly search over design space
        t0 = time.time()
        new_experiment_inds = brnn.fast_search(main_data_scaled, objective, scaler, n_test, P=main_obj_params)
        new_experiments = unique_exp_names[new_experiment_inds]
        elapsed_time.append(time.time()-t0)

        # collect new data 
        new_df   = main_df.iloc[np.in1d(all_exp_names, new_experiments)].copy()
        new_data, new_obj_params, new_experiments, N = format_data(new_df, species, outputs, controls, obj_params=obj_params)

        # remove training samples from main dataset
        train_inds = np.in1d(unique_exp_names, new_experiments)
        main_data = main_data[~train_inds]
        main_obj_params  = main_obj_params[~train_inds]
        unique_exp_names = unique_exp_names[~train_inds]

        # compute objectives
        target_found = []
        for sample, new_obj_param in zip(new_data, new_obj_params):
            #target_found.append(objective(sample[:, :len(sys_var)], new_obj_param))
            target_found.append(objective(sample,new_obj_param))
        target_found = np.array(target_found)

        # store the best objective found (so far)
        objective_found.append(np.max([np.max(objective_found), np.max(target_found)]))

        # Update dataset
        train_df = pd.concat((train_df, new_df))
        train_data, train_obj_params, train_experiments, N = format_data(train_df, species, outputs, controls, obj_params=obj_params)
        print(train_data)

    ### fit model one last time to assess final prediction performance ### 
    # scale train and design space data
    scaler = ZeroMaxScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    main_data_scaled = scaler.transform(main_data)

    # fit model 
    brnn = miRNN(n_species=n_s, n_metabolites=0, n_controls=len(controls), n_hidden=16, N=N)
    brnn.fit(train_data)

    # assess prediction performance of end-point product
    pred, stdv, cov = brnn.predict(main_data_scaled)
    pred = scaler.inverse_transform(pred)
    stdv = scaler.inverse_transform(stdv)
    rvalue = linregress(np.array(main_data[:, -1, 0]), pred[:, -1, -1]).rvalue
    sse_value = sse(np.array(main_data[:, -1, 0]), pred[:, -1, -1])
    plt.scatter(np.array(main_data[:, -1, 0]), pred[:, -1, -1], label="R = {:.3f}\nSSE = {:.3f}".format(rvalue, sse_value))
    plt.legend()
    plt.show()
    objective_rval.append(rvalue)
    objective_sse.append(sse_value)

    plt.scatter(np.array(main_data[:,-1,1]), np.array(main_data[:,-1,0]))
    plt.scatter(np.array(main_data[:,-1,1]), pred[:, -1, -1],color = 'red')
    plt.ylim([0,max(pred[:,-1,-1])+1])
    plt.show()

    '''
    # save data to dataframe
    dtl_df_i = pd.DataFrame()
    dtl_df_i['Trial'] = [trial]
    for j,obj_found in enumerate(objective_found):
        dtl_df_i[f'DTL {j}'] = [obj_found]
    dtl_df = pd.concat((dtl_df, dtl_df_i))
    
    # save data to dataframe
    dtl_df_r = pd.DataFrame()
    dtl_df_r['Trial'] = [trial]
    for j,r_val in enumerate(objective_rval):
        dtl_df_r[f'DTL {j}'] = [r_val]
    dtl_df_R = pd.concat((dtl_df_R, dtl_df_r))
    
    # save data to dataframe
    dtl_df_e = pd.DataFrame()
    dtl_df_e['Trial'] = [trial]
    for j,e in enumerate(objective_sse):
        dtl_df_e[f'DTL {j}'] = [e]
    dtl_df_sse = pd.concat((dtl_df_sse, dtl_df_e))
    
    dtl_df.to_csv("Fig3/miRNN_explore_exploit_FAST.csv", index=False)
    dtl_df_R.to_csv("Fig3/miRNN_explore_exploit_FAST_rvals.csv", index=False)
    dtl_df_sse.to_csv("Fig3/miRNN_explore_exploit_FAST_sse.csv", index=False)
    '''

def Explore_Exploit_1D_csv():
    params = {'legend.fontsize': 18,
          'figure.figsize': (8, 6),
         'axes.labelsize': 24,
         'axes.titlesize':24,
         'axes.linewidth':5,
         'xtick.labelsize':20,
         'ytick.labelsize':20}
    plt.rcParams.update(params)

    plt.rcParams['pdf.fonttype'] = 42
    
    from numpy.random import default_rng
    rng = default_rng(seed = None)# 123)

    # number of trials 
    n_trials = 30

    # number of dtl cycles 
    n_dtl  = 4

    # define number of initial samples to train on
    n_init = 3

    # number of samples for next experiment 
    n_test = 2

    # number of species in model
    n_s = 1

    # number of resources
    n_r = 0

    # define all system variables 
    species = ['Biomass']
    outputs = []
    sys_var = species + outputs

    # define parameters in the objective function
    obj_params = []

    # define subset of controls
    controls = ['Light Intensity']
    system_variables = species + outputs + controls

    # define an objective function (product of endpoint volume * endpoint product concentration)
    # where product concentration is the last column of the predicted output
    time_interval = np.linspace(0,120,11)
    def objective(y,obj):
        return y[-1,0]
        #return trapezoid(y[:,0],time_interval)

    # import data 
    rows_to_skip = list(range(1,350))
    main_df = pd.read_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/BO Algos/Data/Noise_1000_light.csv")#, skiprows = rows_to_skip)
    all_exp_names = main_df.Experiments.values
    # determine random sets of initial experiments
    initial_exps = [rng.choice(np.unique(all_exp_names), n_init, replace=False) for _ in range(n_trials)]
    # function to compute sum of squares error 
    def sse(a, b):
        return np.sum((a-b)**2)

    # init dataframe that stores DTL information
    dtl_df = pd.DataFrame()
    dtl_df_R = pd.DataFrame()
    dtl_df_sse = pd.DataFrame()
    elapsed_time = []

    x_all = []
    y_all = []
    

    for trial in range(n_trials):
        print(trial)

        # format data 
        main_data, main_obj_params, unique_exp_names, N_total = format_data(main_df, species, outputs, controls, obj_params=obj_params)

        # keep track of objective 
        objective_found = []
        x_objective_found = []

        # choose random set of training samples
        train_df = main_df.iloc[np.in1d(all_exp_names, initial_exps[trial])].copy()
        train_data, train_obj_params, new_experiments, N = format_data(train_df, species, outputs, controls, obj_params=obj_params)

        # remove training samples from main dataset so that they're not selected more than once
        train_inds = np.in1d(unique_exp_names, new_experiments)
        main_data = main_data[~train_inds]
        main_obj_params  = main_obj_params[~train_inds]
        unique_exp_names = unique_exp_names[~train_inds]

        # compute objectives
        target_found = []
        x_found = []
        for sample, train_obj_param in zip(train_data, train_obj_params):
            target_found.append(objective(sample[:, :len(sys_var)], train_obj_param))
            x_found.append(sample[0,1])
        target_found = np.array(target_found)
        x_found = np.array(x_found)
        objective_found.append(np.max(target_found))
        x_objective_found.append(x_found[np.argmax(target_found)])
        objective_rval = []
        objective_sse  = []

        # Search over full factorial and update model
        for dtl in range(n_dtl):
            print(f"Running trial {trial+1}, cycle {dtl+1}")

            # scale train and design space data
            scaler = ZeroMaxScaler().fit(train_data)
            train_data = scaler.transform(train_data)
            main_data_scaled = scaler.transform(main_data)

            # fit model 
            brnn = miRNN(n_species=n_s, n_metabolites=0, n_controls=len(controls), n_hidden=16, N=N)
            brnn.fit(train_data)

            # assess prediction performance of end-point product
            pred, stdv, cov = brnn.predict(main_data_scaled)
            pred = scaler.inverse_transform(pred)
            stdv = scaler.inverse_transform(stdv)
            rvalue = linregress(np.array(main_data[:, -1, 0]), pred[:, -1, -1]).rvalue
            sse_value = sse(np.array(main_data[:, -1, 0]), pred[:, -1, -1])
            print(main_data)
            print(pred)
            plt.scatter(np.array(main_data[:, -1, 0]), pred[:, -1, -1], label="R = {:.3f}\nSSE = {:.3f}".format(rvalue, sse_value))
            plt.legend()
            plt.show()
            '''
            
            #Prove initial training set can identify correct trend
            fig, axs = plt.subplots(2,3)
            step = int((len(main_data)-1)/6)
            
            
            # This displays time series data uniform across the light intensities
            
            for i in range(6):
                rand = step*(i+1)

                axs[int(5/3-i/3),i%3].scatter(time_interval,main_data[rand,:,0],label = 'Data: ' + str(main_data[rand,0,-1]))
                axs[int(5/3-i/3),i%3].plot(time_interval,pred[rand,:,-1],label = 'Predicted')
                axs[int(5/3-i/3),i%3].legend()
            
            fig.text(0.08, 0.5, 'Biomass Concentration', va='center', rotation='vertical',fontsize='20')
            fig.text(.5, .04, 'Time', va='center',fontsize = '20')
            plt.suptitle('Time Series @ Different Light Intensities: Sigmoid',fontsize = '28')
            plt.show()
            

            plt.scatter(main_data[:,-1,1],main_data[:,-1,0], label = 'Actual')
            plt.scatter(main_data[:,-1,1],pred[:,-1,-1], label = 'Predicted')
            
            plt.title('Final Biomass Concentration across all light intensities, Sigmoid',fontsize = '14')
            plt.ylabel('Final Biomass Concentration',fontsize = '14')
            plt.xlabel('Light Intensity',fontsize = '14')
            plt.legend()
            plt.show()
            '''

            objective_rval.append(rvalue)
            objective_sse.append(sse_value)
            
            # randomly search over design space
            t0 = time.time()
            new_experiment_inds = brnn.fast_search(main_data_scaled, objective, scaler, n_test, P=main_obj_params)
            new_experiments = unique_exp_names[new_experiment_inds]
            elapsed_time.append(time.time()-t0)

            # collect new data 
            new_df   = main_df.iloc[np.in1d(all_exp_names, new_experiments)].copy()
            new_data, new_obj_params, new_experiments, N = format_data(new_df, species, outputs, controls, obj_params=obj_params)
            
            # remove training samples from main dataset
            train_inds = np.in1d(unique_exp_names, new_experiments)
            main_data = main_data[~train_inds]
            main_obj_params  = main_obj_params[~train_inds]
            unique_exp_names = unique_exp_names[~train_inds]

            # compute objectives
            target_found = []
            x_found = []
            ind_vars = []
            for sample, new_obj_param in zip(new_data, new_obj_params):
                target_found.append(objective(sample[:, :len(sys_var)], new_obj_param))
                x_found.append(sample[0,1])

            target_found = np.array(target_found)
            x_found = np.array(x_found)

            # store the best objective found (so far)
            if (np.max(objective_found) > np.max(target_found)):
                x_objective_found.append(x_objective_found[np.argmax(objective_found)])
            else:
                x_objective_found.append(x_found[np.argmax(x_found)])
                
            objective_found.append(np.max([np.max(objective_found), np.max(target_found)]))
            

            # Update dataset
            train_df = pd.concat((train_df, new_df))
            train_data, train_obj_params, train_experiments, N = format_data(train_df, species, outputs, controls, obj_params=obj_params)

        ### fit model one last time to assess final prediction performance ### 
        # scale train and design space data
            
        print(objective_found)
        print(x_objective_found)

        if len(y_all) == 0:
            y_all = objective_found
            x_all = x_objective_found
        else:
            y_all = np.vstack([y_all, objective_found])
            x_all = np.vstack([x_all, x_objective_found])
        continue

        scaler = ZeroMaxScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        main_data_scaled = scaler.transform(main_data)

        # fit model 
        brnn = miRNN(n_species=n_s, n_metabolites=0, n_controls=len(controls), n_hidden=16, N=N)
        brnn.fit(train_data)

        # assess prediction performance of end-point product
        pred, stdv, cov = brnn.predict(main_data_scaled)
        pred = scaler.inverse_transform(pred)
        stdv = scaler.inverse_transform(stdv)
        rvalue = linregress(np.array(main_data[:, -1, 0]), pred[:, -1, -1]).rvalue
        sse_value = sse(np.array(main_data[:, -1, 0]), pred[:, -1, -1])
        plt.scatter(np.array(main_data[:, -1, 0]), pred[:, -1, -1], label="R = {:.3f}\nSSE = {:.3f}".format(rvalue, sse_value))
        plt.legend()
        plt.show()
        objective_rval.append(rvalue)
        objective_sse.append(sse_value)

        # printing data
        print(objective_found)
        '''
        # save data to dataframe
        dtl_df_i = pd.DataFrame()
        dtl_df_i['Trial'] = [trial]
        for j,obj_found in enumerate(objective_found):
            dtl_df_i[f'DTL {j}'] = [obj_found]
        dtl_df = pd.concat((dtl_df, dtl_df_i))
        
        # save data to dataframe
        dtl_df_r = pd.DataFrame()
        dtl_df_r['Trial'] = [trial]
        for j,r_val in enumerate(objective_rval):
            dtl_df_r[f'DTL {j}'] = [r_val]
        dtl_df_R = pd.concat((dtl_df_R, dtl_df_r))
        
        # save data to dataframe
        dtl_df_e = pd.DataFrame()
        dtl_df_e['Trial'] = [trial]
        for j,e in enumerate(objective_sse):
            dtl_df_e[f'DTL {j}'] = [e]
        dtl_df_sse = pd.concat((dtl_df_sse, dtl_df_e))
        
        dtl_df.to_csv("Fig3/miRNN_explore_exploit_FAST.csv", index=False)
        dtl_df_R.to_csv("Fig3/miRNN_explore_exploit_FAST_rvals.csv", index=False)
        dtl_df_sse.to_csv("Fig3/miRNN_explore_exploit_FAST_sse.csv", index=False)
        '''
    
    all = pd.DataFrame(np.hstack([y_all,x_all]))
    all.to_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/BO Algos/Data/Noise_" + str(n_init) +"_intial_" + str(n_test) + "_samp_"+str(n_dtl)+"_rounds_"+str(n_trials)+".csv",index = False)

def Explore_Exploit_2D_csv():

    params = {'legend.fontsize': 18,
          'figure.figsize': (8, 6),
         'axes.labelsize': 24,
         'axes.titlesize':24,
         'axes.linewidth':5,
         'xtick.labelsize':20,
         'ytick.labelsize':20}
    plt.rcParams.update(params)

    plt.rcParams['pdf.fonttype'] = 42
    
    from numpy.random import default_rng
    rng = default_rng(seed = None)# 123)

    # number of trials 
    n_trials = 1

    # number of dtl cycles 
    n_dtl  = 4

    # define number of initial samples to train on
    n_init = 3

    # number of samples for next experiment 
    n_test = 3

    # number of species in model
    n_s = 1

    # number of resources
    n_r = 0

    # define all system variables 
    species = ['Biomass']
    outputs = []
    sys_var = species + outputs

    # define parameters in the objective function
    obj_params = []

    # define subset of controls
    controls = ['Light Intensity','Nitrogen']
    system_variables = species + outputs + controls

    # define an objective function (product of endpoint volume * endpoint product concentration)
    # where product concentration is the last column of the predicted output
    time_interval = np.linspace(0,200,10)
    def objective(y,obj):
        return y[-1,0]
        #return trapezoid(y[:,0],time_interval)

    # import data 
    rows_to_skip = list(range(1,350))
    main_df = pd.read_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/BO Algos/Data/2000_light_400_Nitrogen.csv")#, skiprows = rows_to_skip)
    all_exp_names = main_df.Experiments.values
    # determine random sets of initial experiments
    initial_exps = [rng.choice(np.unique(all_exp_names), n_init, replace=False) for _ in range(n_trials)]
    # function to compute sum of squares error 
    def sse(a, b):
        return np.sum((a-b)**2)

    # init dataframe that stores DTL information
    dtl_df = pd.DataFrame()
    dtl_df_R = pd.DataFrame()
    dtl_df_sse = pd.DataFrame()
    elapsed_time = []

    x_all = []
    y_all = []
    
    for trial in range(n_trials):
        print(trial)

        # format data 
        main_data, main_obj_params, unique_exp_names, N_total = format_data(main_df, species, outputs, controls, obj_params=obj_params)

        # keep track of objective 
        objective_found = []
        x_objective_found = []

        # choose random set of training samples
        train_df = main_df.iloc[np.in1d(all_exp_names, initial_exps[trial])].copy()
        train_data, train_obj_params, new_experiments, N = format_data(train_df, species, outputs, controls, obj_params=obj_params)

        # remove training samples from main dataset so that they're not selected more than once
        train_inds = np.in1d(unique_exp_names, new_experiments)
        main_data = main_data[~train_inds]
        main_obj_params  = main_obj_params[~train_inds]
        unique_exp_names = unique_exp_names[~train_inds]

        # compute objectives
        target_found = []
        x_found = []
        for sample, train_obj_param in zip(train_data, train_obj_params):
            #Appending dependent variable
            target_found.append(objective(sample[:, :len(sys_var)], train_obj_param)) #Change this to BO_call
            #Appending the 2 independent variables
            x_found.append(sample[0,len(sys_var):])
        
        target_found = np.array(target_found)
        x_found = np.array(x_found)
        objective_found.append(np.max(target_found))
        x_objective_found.append(x_found[np.argmax(target_found)])
        objective_rval = []
        objective_sse  = []

        # Search over full factorial and update model
        for dtl in range(n_dtl):
            print(f"Running trial {trial+1}, cycle {dtl+1}")

            # scale train and design space data
            scaler = ZeroMaxScaler().fit(train_data)
            train_data = scaler.transform(train_data)
            main_data_scaled = scaler.transform(main_data)

            # fit model 
            brnn = miRNN(n_species=n_s, n_metabolites=0, n_controls=len(controls), n_hidden=16, N=N)
            brnn.fit(train_data)

            # assess prediction performance of end-point product
            pred, stdv, cov = brnn.predict(main_data_scaled)
            pred = scaler.inverse_transform(pred)
            stdv = scaler.inverse_transform(stdv)

            print(main_data[:, -1, 0])
            print(pred[:,-1])
            print(pred[-1,:].reshape(1,-1))     

            print(len(main_data[:, -1, 0]))
            print(len(pred[-1,:].reshape(1,-1)))
            print(len(pred[:,-1]))
            
            rvalue = linregress(np.array(main_data[:, -1, 0]), pred[:,-1].reshape(1,-1)).rvalue                                                                                                                                                                   
            sse_value = sse(np.array(main_data[:, -1, 0]), pred[:,-1].reshape(1,-1))
            '''
            plt.scatter(np.array(main_data[:, -1, 0]), pred[:,-1].reshape(1,-1), label="R = {:.3f}\nSSE = {:.3f}".format(rvalue, sse_value))
            plt.legend()
            plt.show()
            
            
            #Prove initial training set can identify correct trend
            fig, axs = plt.subplots(2,3)
            step = int((len(main_data)-1)/6)
            
            
            # This displays time series data uniform across the light intensities
            
            for i in range(6):
                rand = step*(i+1)

                axs[int(5/3-i/3),i%3].scatter(time_interval,main_data[rand,:,0],label = 'Data: ' + str(main_data[rand,0,-1]))
                axs[int(5/3-i/3),i%3].plot(time_interval,pred[rand,:,-1],label = 'Predicted')
                axs[int(5/3-i/3),i%3].legend()
            
            fig.text(0.08, 0.5, 'Biomass Concentration', va='center', rotation='vertical',fontsize='20')
            fig.text(.5, .04, 'Time', va='center',fontsize = '20')
            plt.suptitle('Time Series @ Different Light Intensities: Sigmoid',fontsize = '28')
            plt.show()
            
            plt.scatter(main_data[:,-1,1],main_data[:,-1,0], label = 'Actual')
            plt.scatter(main_data[:,-1,1],pred[:,-1,-1], label = 'Predicted')
            
            plt.title('Final Biomass Concentration across all light intensities, Sigmoid',fontsize = '14')
            plt.ylabel('Final Biomass Concentration',fontsize = '14')
            plt.xlabel('Light Intensity',fontsize = '14')
            plt.legend()
            plt.show()
            '''

            objective_rval.append(rvalue)
            objective_sse.append(sse_value)
            
            # randomly search over design space
            new_experiment_inds = brnn.fast_search(main_data_scaled, objective, scaler, n_test, P=main_obj_params)
            new_experiments = unique_exp_names[new_experiment_inds]

            # collect new data 
            new_df   = main_df.iloc[np.in1d(all_exp_names, new_experiments)].copy()
            new_data, new_obj_params, new_experiments, N = format_data(new_df, species, outputs, controls, obj_params=obj_params)
            
            # remove training samples from main dataset
            train_inds = np.in1d(unique_exp_names, new_experiments)
            main_data = main_data[~train_inds]
            main_obj_params  = main_obj_params[~train_inds]
            unique_exp_names = unique_exp_names[~train_inds]

            # compute objectives
            target_found = []
            x_found = []
            ind_vars = []
            for sample, new_obj_param in zip(new_data, new_obj_params):
                #Appending the objective value
                target_found.append(objective(sample[:, :len(sys_var)], new_obj_param))
                #Appending the 2 depedent variables
                x_found.append(sample[0,1:])

            target_found = np.array(target_found)
            x_found = np.array(x_found)
        

            # store the best objective found (so far)
            if (np.max(objective_found) > np.max(target_found)):
                x_objective_found.append(x_objective_found[np.argmax(objective_found)])
            else:
                print(np.argmax(x_found))
                print(x_found)
                
                x_objective_found.append(x_found[np.argmax(target_found)])
            
            objective_found.append(np.max([np.max(objective_found), np.max(target_found)]))
            print('found\n' + str(x_found))
            print('x_obj\n' + str(x_objective_found))
            print('y_found\n' + str(target_found))
            print('obj\n' + str(objective_found))
            print(str(np.array(x_objective_found)[:,0]) + ' & ' + str(np.array(x_objective_found)[:,1]))
            '''
            plt.scatter(np.array(x_objective_found)[:,0],np.array(x_objective_found)[:,1])
            plt.show()
            '''

            # Update dataset
            train_df = pd.concat((train_df, new_df))
            train_data, train_obj_params, train_experiments, N = format_data(train_df, species, outputs, controls, obj_params=obj_params)

        ### fit model one last time to assess final prediction performance ### 
        # scale train and design space data
           
        if len(y_all) == 0:
            y_all = objective_found
            x_all = np.array(x_objective_found).reshape(1,-1)
        else:
            y_all = np.vstack([y_all, objective_found])
            x_all = np.vstack([x_all, np.array(x_objective_found).reshape(1,-1)])
        
        scaler = ZeroMaxScaler().fit(train_data)
        train_data = scaler.transform(train_data)
        main_data_scaled = scaler.transform(main_data)

        # fit model 
        brnn = miRNN(n_species=n_s, n_metabolites=0, n_controls=len(controls), n_hidden=16, N=N)
        brnn.fit(train_data)
        
        # assess prediction performance of end-point product
        pred, stdv, cov = brnn.predict(main_data_scaled)
        pred = scaler.inverse_transform(pred)
        stdv = scaler.inverse_transform(stdv)
        rvalue = linregress(np.array(main_data[:, -1, 0]), pred[:,-1].reshape(1,-1)).rvalue
        sse_value = sse(np.array(main_data[:, -1, 0]), pred[:,-1].reshape(1,-1))
        plt.scatter(np.array(main_data[:, -1, 0]), pred[:,-1].reshape(1,-1), label="R = {:.3f}\nSSE = {:.3f}".format(rvalue, sse_value))
        
        plt.legend()
        plt.show()
        objective_rval.append(rvalue)
        objective_sse.append(sse_value)
        '''
        plt.scatter(np.array(x_objective_found)[:,0],np.array(x_objective_found)[:,1])
        plt.show()
        '''
        '''
        # save data to dataframe
        dtl_df_i = pd.DataFrame()
        dtl_df_i['Trial'] = [trial]
        for j,obj_found in enumerate(objective_found):
            dtl_df_i[f'DTL {j}'] = [obj_found]
        dtl_df = pd.concat((dtl_df, dtl_df_i))
        
        # save data to dataframe
        dtl_df_r = pd.DataFrame()
        dtl_df_r['Trial'] = [trial]
        for j,r_val in enumerate(objective_rval):
            dtl_df_r[f'DTL {j}'] = [r_val]
        dtl_df_R = pd.concat((dtl_df_R, dtl_df_r))
        
        # save data to dataframe
        dtl_df_e = pd.DataFrame()
        dtl_df_e['Trial'] = [trial]
        for j,e in enumerate(objective_sse):
            dtl_df_e[f'DTL {j}'] = [e]
        dtl_df_sse = pd.concat((dtl_df_sse, dtl_df_e))
        
        dtl_df.to_csv("Fig3/miRNN_explore_exploit_FAST.csv", index=False)
        dtl_df_R.to_csv("Fig3/miRNN_explore_exploit_FAST_rvals.csv", index=False)
        dtl_df_sse.to_csv("Fig3/miRNN_explore_exploit_FAST_sse.csv", index=False)
        '''
    test_algae_BO_2D_Nitrogen()
    print(train_data)
    plt.title('Parametric')
    plt.scatter(x_all[:,-2],x_all[:,-1],color = 'blue',zorder=6,label = 'Maximum')
    #plt.scatter(scaler.inverse_transform(train_data)[:, -1, 1], scaler.inverse_transform(train_data)[:, -1, 2],zorder=5)
    plt.xlabel('Light Intensity')
    plt.ylabel('Nitrogen Concentration')
    print(x_all)
    print(y_all)
    plt.show()
    all = pd.DataFrame(np.hstack([y_all,x_all]))
    all.to_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/BO Algos/Data/2D_" + str(n_init) +"_intial_" + str(n_test) + "_samp_"+str(n_dtl)+"_rounds_"+str(n_trials)+".csv",index = False)
    
    print('asdfasdf')

def Explore_Exploit_5D():
    start0 = time.time()
    time_results = []
    time_results.append(start0)
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

    n_hidden = 4

    kernel = []
    exp_w = []
    t_eval = 11

    BO_obj = myBO.BO(ub,lb,step,dim = dim,exp_w = exp_w,kernel = kernel,system = PM_5.BO_intializer,bounds = bnds)  
    org_t = time.time()
    BO_obj.auto_create_axis()
    BO_obj.iterate(0)

    main_conditions = BO_obj.descale(BO_obj.grid_vertices)
    row_len = len(main_conditions)
    params = {'legend.fontsize': 18,
          'figure.figsize': (8, 6),
         'axes.labelsize': 24,
         'axes.titlesize':24,
         'axes.linewidth':5,
         'xtick.labelsize':20,
         'ytick.labelsize':20}
    plt.rcParams.update(params)

    plt.rcParams['pdf.fonttype'] = 42
    
    from numpy.random import default_rng
    rng = default_rng(seed =  543)

    # number of trials 
    n_trials = 10

    # number of dtl cycles 
    n_dtl  = 5

    # define number of initial samples to train on
    n_init = 10

    # number of samples for next experiment 
    n_test = 40

    # number of species in model
    n_s = 1

    # number of resources
    n_r = 0

    # define all system variables 
    species = ['Biomass']
    outputs = []
    sys_var = species + outputs

    # define parameters in the objective function
    obj_params = []

    # define subset of controls
    controls = ['Light Intensity','Nitrogen','Quota','Light Path','Sun Time']
    system_variables = species + outputs + controls

    # define an objective function (product of endpoint volume * endpoint product concentration)
    # where product concentration is the last column of the predicted output
    
    def objective(data, obj_params = 0): ###TODO This will need to be edited to call function, maybe
        return data[-1,0]
        #return trapezoid(y[:,0],time_interval)
    
    def objective_time_collect(data):
        '''
        Fills in the train_data with biomass over time
        '''
        obj = data[:,-1,1:]

        for row in range(len(obj)):
            y = PM_5.it(obj[row,:])[0,:]
            data[row,:,0] = y

        return data

    def main_expansion(data):
        '''
        This expands a matrix of coordinates to have the time dimension. This is needed for the RNN 
        '''
        num_rows,num_cols = data.shape
        n = 11
        zeros = np.zeros((num_rows, n, num_cols))
        # Stack the original matrix and the zeros along the new axis
        new_matrix = np.zeros((num_rows, 11, num_cols + 1))

        # Fill the new 3D matrix
        for i in range(num_rows):
            #Fill in conditions across time
            new_matrix[i,:,1:] = np.tile(data[i, :], (11, 1))
            #Give initial Biomass start
            new_matrix[i,0,0] = 0.0555

        return new_matrix

    main_data = main_expansion(main_conditions)

    # import data ## TODO Delete this and replace with list of grid arguments
    # determine random sets of initial experiments
    initial_exps = [rng.choice(row_len, n_init, replace=False) for _ in range(n_trials)]
    
    compare_ind =  rng.choice(row_len, 10000, replace=False)
    
    main_data[compare_ind,:,:] = objective_time_collect(main_data[compare_ind,:,:])
    # function to compute sum of squares error 
    def sse(a, b):
        return np.sum((a-b)**2)

    # init dataframe that stores DTL information
    dtl_df = pd.DataFrame()
    dtl_df_R = pd.DataFrame()
    dtl_df_sse = pd.DataFrame()
    elapsed_time = []

    x_all = []
    y_all = []
    time_array = np.zeros((n_trials,n_dtl))
    
    objective_rval = []
    objective_sse  = []
    
    for trial in range(n_trials):
        time_results.append(time.time())
        print('trial' + str(trial+1))
        if trial == 1:
            continue

        # choose random set of training samples ##TODO edit to make avaliable for a Numpy matirx
        train_data = objective_time_collect(main_data[initial_exps[trial]])

        # keep track of objective 
        objective_found = []
        x_objective_found = []

        
        #train_data, train_obj_params, new_experiments, N = format_data(train_df, species, outputs, controls, obj_params=obj_params)

        # remove training samples from main dataset so that they're not selected more than once ##TODO edit to make avaliable for a Numpy matirx
        '''
        mask = np.ones(main_data.shape[0], dtype=bool)
        mask[initial_exps[trial]] = False
        #main_data = main_data[mask]
        print('1.5')
        m = []
        for batch_inds in np.array_split(np.arange(len(mask)), len(mask)//512):
            m.append(main_data[batch_inds])
        main_data = m
        '''

        # compute objectives ###TODO May need to edit for-loop
        target_found = []
        x_found = []

        for sample in train_data:
            #Appending the objective value
            target_found.append(objective(sample))
            #Appending the 2 depedent variables
            x_found.append(sample[0,1:])

        target_found = np.array(target_found)
        x_found = np.array(x_found)
        objective_found.append(np.max(target_found))
        x_objective_found.append(x_found[np.argmax(target_found)])
        
        # Search over full factorial and update model
        starting_time = time.time()
        for dtl in range(n_dtl):
            print(f"Running trial {trial+1}, cycle {dtl+1}")

            scaler = ZeroMaxScaler().fit(train_data)
            train_data_scaled = scaler.transform(train_data)
            main_data_scaled = scaler.transform(main_data)

            # fit model 
            N = len(train_data) * (t_eval - 1)
            brnn = miRNN(n_species=n_s, n_metabolites=0, n_controls=len(controls), n_hidden=n_hidden, N=N)
            brnn.fit(train_data_scaled,trial = trial)  ###Check Train_data format

            # assess prediction performance of end-point product
            pred, stdv, cov = brnn.predict(main_data_scaled)  ##Check main_data format
            pred = scaler.inverse_transform(pred)
            stdv = scaler.inverse_transform(stdv)
            
            rvalue = linregress(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1)).rvalue                                                                                                                                                                   
            sse_value = sse(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1))
            plt.scatter(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1), label="R = {:.3f}\nSSE = {:.3f}".format(rvalue, sse_value))
            plt.legend()
            #plt.show()
            
            objective_rval.append(rvalue)
            objective_sse.append(sse_value)
            '''
            plt.scatter(np.array(main_data[:, -1, 0]), pred[:,-1].reshape(1,-1), label="R = {:.3f}\nSSE = {:.3f}".format(rvalue, sse_value))
            plt.legend()
            plt.show()
            
            #Prove initial training set can identify correct trend
            fig, axs = plt.subplots(2,3)
            step = int((len(main_data)-1)/6)
            
            
            # This displays time series data uniform across the light intensities
            
            for i in range(6):
                rand = step*(i+1)

                axs[int(5/3-i/3),i%3].scatter(time_interval,main_data[rand,:,0],label = 'Data: ' + str(main_data[rand,0,-1]))
                axs[int(5/3-i/3),i%3].plot(time_interval,pred[rand,:,-1],label = 'Predicted')
                axs[int(5/3-i/3),i%3].legend()
            
            fig.text(0.08, 0.5, 'Biomass Concentration', va='center', rotation='vertical',fontsize='20')
            fig.text(.5, .04, 'Time', va='center',fontsize = '20')
            plt.suptitle('Time Series @ Different Light Intensities: Sigmoid',fontsize = '28')
            plt.show()
            
            plt.scatter(main_data[:,-1,1],main_data[:,-1,0], label = 'Actual')
            plt.scatter(main_data[:,-1,1],pred[:,-1,-1], label = 'Predicted')
            
            plt.title('Final Biomass Concentration across all light intensities, Sigmoid',fontsize = '14')
            plt.ylabel('Final Biomass Concentration',fontsize = '14')
            plt.xlabel('Light Intensity',fontsize = '14')
            plt.legend()
            plt.show()
            '''

            # randomly search over design space
            new_experiment_inds = brnn.fast_search(main_data_scaled, objective, scaler, n_test, P=main_data)   ####TODO: ensure this works find without y-value in main_data_scaled
            new_experiments = main_data[new_experiment_inds]
            
            # remove training samples from main dataset  ###TODO: change indexing maybe
            #main_data = main_data[~new_experiment_inds]

            new_experiments = objective_time_collect(new_experiments)

            # compute objectives   
            target_found = []
            x_found = []
            ind_vars = []

            for sample in new_experiments:
                #Appending the objective value
                target_found.append(objective(sample))
                #Appending the 2 depedent variables
                x_found.append(sample[0,1:])

            target_found = np.array(target_found)
            x_found = np.array(x_found)
        
            # store the best objective found (so far)
            if (np.max(objective_found) > np.max(target_found)):
                x_objective_found.append(x_objective_found[np.argmax(objective_found)])
            else:
                x_objective_found.append(x_found[np.argmax(target_found)])
            
            objective_found.append(np.max([np.max(objective_found), np.max(target_found)]))
            '''
            print('found\n' + str(x_found))
            print('x_obj\n' + str(x_objective_found))
            print('y_found\n' + str(target_found))
            print('obj\n' + str(objective_found))
            '''
            #print(str(np.array(x_objective_found)[:,0]) + ' & ' + str(np.array(x_objective_found)[:,1]))
            '''
            plt.scatter(np.array(x_objective_found)[:,0],np.array(x_objective_found)[:,1])
            plt.show()
            '''
  
            train_data = np.vstack([train_data,new_experiments])
            time_array[trial,dtl] = time.time() - starting_time
            starting_time = time.time()

        ### fit model one last time to assess final prediction performance ### 
        # scale train and design space data
           
        if trial == 0:
            y_all = objective_found
            x_all = np.array(x_objective_found).reshape(1,-1)
        else:
            y_all = np.vstack([y_all, objective_found])
            x_all = np.vstack([x_all, np.array(x_objective_found).reshape(1,-1)])
        
        scaler = ZeroMaxScaler().fit(train_data)
        train_data_scaled = scaler.transform(train_data)
        main_data_scaled = scaler.transform(main_data)

        # fit model 
        N = len(train_data) * (t_eval - 1)
        brnn = miRNN(n_species=n_s, n_metabolites=0, n_controls=len(controls), n_hidden=n_hidden, N=N)
        brnn.fit(train_data_scaled)
        
        # assess prediction performance of end-point product
        pred, stdv, cov = brnn.predict(main_data_scaled)
        pred = scaler.inverse_transform(pred)
        stdv = scaler.inverse_transform(stdv)
        
        rvalue = linregress(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1)).rvalue
        sse_value = sse(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1))
        plt.scatter(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1), label="R = {:.3f}\nSSE = {:.3f}".format(rvalue, sse_value))
        
        plt.legend()
        #plt.show()
        objective_rval.append(rvalue)
        objective_sse.append(sse_value)
        
        '''
        plt.scatter(np.array(x_objective_found)[:,0],np.array(x_objective_found)[:,1])
        plt.show()
        '''
        '''
        # save data to dataframe
        dtl_df_i = pd.DataFrame()
        dtl_df_i['Trial'] = [trial]
        for j,obj_found in enumerate(objective_found):
            dtl_df_i[f'DTL {j}'] = [obj_found]
        dtl_df = pd.concat((dtl_df, dtl_df_i))
        
        # save data to dataframe
        dtl_df_r = pd.DataFrame()
        dtl_df_r['Trial'] = [trial]
        for j,r_val in enumerate(objective_rval):
            dtl_df_r[f'DTL {j}'] = [r_val]
        dtl_df_R = pd.concat((dtl_df_R, dtl_df_r))
        
        # save data to dataframe
        dtl_df_e = pd.DataFrame()
        dtl_df_e['Trial'] = [trial]
        for j,e in enumerate(objective_sse):
            dtl_df_e[f'DTL {j}'] = [e]
        dtl_df_sse = pd.concat((dtl_df_sse, dtl_df_e))
        
        dtl_df.to_csv("Fig3/miRNN_explore_exploit_FAST.csv", index=False)
        dtl_df_R.to_csv("Fig3/miRNN_explore_exploit_FAST_rvals.csv", index=False)
        dtl_df_sse.to_csv("Fig3/miRNN_explore_exploit_FAST_sse.csv", index=False)
        '''

    time_results.append(time.time())
    print(train_data)
    print(x_all)
    print(len(x_all))
    print(y_all)
    print(len(y_all))

    all = pd.DataFrame(np.hstack([y_all,x_all]))
    all.to_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/5-D/RNN/5D_" + str(n_init) +"_intial_" + str(n_test) + "_samp_"+str(n_dtl)+"_rounds_"+str(n_trials)+"_tanh_" + str(n_hidden) + ".csv",index = False)

    ti = pd.DataFrame(time_results)
    ti.to_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/5-D/RNN/5D_time_" + str(n_init) +"_intial_" + str(n_test) + "_samp_"+str(n_dtl)+"_rounds_"+str(n_trials)+"_tanh_" + str(n_hidden) + ".csv",index = False)

    r = pd.DataFrame(np.hstack([objective_rval,objective_sse]))
    r.to_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/5-D/RNN/5D_Rvals_" + str(n_init) +"_intial_" + str(n_test) + "_samp_"+str(n_dtl)+"_rounds_"+str(n_trials)+"_tanh_" + str(n_hidden) + ".csv",index = False)
    print('asdfasdf')
    
def Explore_Exploit_4D():
    start0 = time.time()
    time_results = []
    time_results.append(start0)
    ####Initializes Grid Creation
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

    kernel = []
    exp_w = []
    t_eval = 11

    BO_obj = myBO.BO(ub,lb,step,dim = dim,exp_w = exp_w,kernel = kernel,system = PM_4.BO_intializer,bounds = bnds)  
    org_t = time.time()
    BO_obj.auto_create_axis()
    BO_obj.iterate(0)

    main_conditions = BO_obj.descale(BO_obj.grid_vertices)
    row_len = len(main_conditions)
    params = {'legend.fontsize': 18,
          'figure.figsize': (8, 6),
         'axes.labelsize': 24,
         'axes.titlesize':24,
         'axes.linewidth':5,
         'xtick.labelsize':20,
         'ytick.labelsize':20}
    plt.rcParams.update(params)

    plt.rcParams['pdf.fonttype'] = 42
    
    from numpy.random import default_rng
    rng = default_rng(seed =  543)

    # number of trials 
    n_trials = 10

    # number of dtl cycles 
    n_dtl  = 5

    # define number of initial samples to train on
    n_init = 10

    # number of samples for next experiment 
    n_test = 40

    # number of species in model
    n_s = 1

    # number of resources
    n_r = 0

    # define all system variables 
    species = ['Biomass']
    outputs = []
    sys_var = species + outputs

    # define parameters in the objective function
    obj_params = []

    # define subset of controls
    controls = ['Light Intensity','Nitrogen','Quota','Sun Time']
    system_variables = species + outputs + controls

    # define an objective function (product of endpoint volume * endpoint product concentration)
    # where product concentration is the last column of the predicted output
    
    def objective(data, obj_params = 0): ###TODO This will need to be edited to call function, maybe
        return data[-1,0]
        #return trapezoid(y[:,0],time_interval)
    
    def objective_time_collect(data):
        '''
        Fills in the train_data with biomass over time
        '''
        obj = data[:,-1,1:]

        for row in range(len(obj)):
            y = PM_4.it(obj[row,:])[0,:]
            data[row,:,0] = y

        return data

    def main_expansion(data):
        '''
        This expands a matrix of coordinates to have the time dimension. This is needed for the RNN 
        '''
        num_rows,num_cols = data.shape
        n = 11
        zeros = np.zeros((num_rows, n, num_cols))
        # Stack the original matrix and the zeros along the new axis
        new_matrix = np.zeros((num_rows, 11, num_cols + 1))

        # Fill the new 3D matrix
        for i in range(num_rows):
            #Fill in conditions across time
            new_matrix[i,:,1:] = np.tile(data[i, :], (11, 1))
            #Give initial Biomass start
            new_matrix[i,0,0] = 0.0555

        return new_matrix

    main_data = main_expansion(main_conditions)

    # import data ## TODO Delete this and replace with list of grid arguments
    # determine random sets of initial experiments
    initial_exps = [rng.choice(row_len, n_init, replace=False) for _ in range(n_trials)]
    
    compare_ind =  rng.choice(row_len, 10000, replace=False)
    
    main_data[compare_ind,:,:] = objective_time_collect(main_data[compare_ind,:,:])
    # function to compute sum of squares error 
    def sse(a, b):
        return np.sum((a-b)**2)

    # init dataframe that stores DTL information
    dtl_df = pd.DataFrame()
    dtl_df_R = pd.DataFrame()
    dtl_df_sse = pd.DataFrame()
    elapsed_time = []

    x_all = []
    y_all = []
    time_array = np.zeros((n_trials,n_dtl))
    
    objective_rval = []
    objective_sse  = []
    
    for trial in range(n_trials):
        time_results.append(time.time())
        print('trial' + str(trial+1))
        if trial == 1:
            continue

        # choose random set of training samples ##TODO edit to make avaliable for a Numpy matirx
        train_data = objective_time_collect(main_data[initial_exps[trial]])

        # keep track of objective 
        objective_found = []
        x_objective_found = []

        
        #train_data, train_obj_params, new_experiments, N = format_data(train_df, species, outputs, controls, obj_params=obj_params)

        # remove training samples from main dataset so that they're not selected more than once ##TODO edit to make avaliable for a Numpy matirx
        '''
        mask = np.ones(main_data.shape[0], dtype=bool)
        mask[initial_exps[trial]] = False
        #main_data = main_data[mask]
        print('1.5')
        m = []
        for batch_inds in np.array_split(np.arange(len(mask)), len(mask)//512):
            m.append(main_data[batch_inds])
        main_data = m
        '''

        # compute objectives ###TODO May need to edit for-loop
        target_found = []
        x_found = []

        for sample in train_data:
            #Appending the objective value
            target_found.append(objective(sample))
            #Appending the 2 depedent variables
            x_found.append(sample[0,1:])

        target_found = np.array(target_found)
        x_found = np.array(x_found)
        objective_found.append(np.max(target_found))
        x_objective_found.append(x_found[np.argmax(target_found)])
        
        # Search over full factorial and update model
        starting_time = time.time()
        for dtl in range(n_dtl):
            print(f"Running trial {trial+1}, cycle {dtl+1}")

            scaler = ZeroMaxScaler().fit(train_data)
            train_data_scaled = scaler.transform(train_data)
            main_data_scaled = scaler.transform(main_data)

            # fit model 
            N = len(train_data) * (t_eval - 1)
            brnn = miRNN(n_species=n_s, n_metabolites=0, n_controls=len(controls), n_hidden=16, N=N)
            brnn.fit(train_data_scaled,trial = trial)  ###Check Train_data format

            # assess prediction performance of end-point product
            pred, stdv, cov = brnn.predict(main_data_scaled)  ##Check main_data format
            pred = scaler.inverse_transform(pred)
            stdv = scaler.inverse_transform(stdv)
            
            rvalue = linregress(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1)).rvalue                                                                                                                                                                   
            sse_value = sse(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1))
            plt.scatter(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1), label="R = {:.3f}\nSSE = {:.3f}".format(rvalue, sse_value))
            plt.legend()
            #plt.show()
            
            objective_rval.append(rvalue)
            objective_sse.append(sse_value)
            '''
            plt.scatter(np.array(main_data[:, -1, 0]), pred[:,-1].reshape(1,-1), label="R = {:.3f}\nSSE = {:.3f}".format(rvalue, sse_value))
            plt.legend()
            plt.show()
            
            #Prove initial training set can identify correct trend
            fig, axs = plt.subplots(2,3)
            step = int((len(main_data)-1)/6)
            
            
            # This displays time series data uniform across the light intensities
            
            for i in range(6):
                rand = step*(i+1)

                axs[int(5/3-i/3),i%3].scatter(time_interval,main_data[rand,:,0],label = 'Data: ' + str(main_data[rand,0,-1]))
                axs[int(5/3-i/3),i%3].plot(time_interval,pred[rand,:,-1],label = 'Predicted')
                axs[int(5/3-i/3),i%3].legend()
            
            fig.text(0.08, 0.5, 'Biomass Concentration', va='center', rotation='vertical',fontsize='20')
            fig.text(.5, .04, 'Time', va='center',fontsize = '20')
            plt.suptitle('Time Series @ Different Light Intensities: Sigmoid',fontsize = '28')
            plt.show()
            
            plt.scatter(main_data[:,-1,1],main_data[:,-1,0], label = 'Actual')
            plt.scatter(main_data[:,-1,1],pred[:,-1,-1], label = 'Predicted')
            
            plt.title('Final Biomass Concentration across all light intensities, Sigmoid',fontsize = '14')
            plt.ylabel('Final Biomass Concentration',fontsize = '14')
            plt.xlabel('Light Intensity',fontsize = '14')
            plt.legend()
            plt.show()
            '''

            # randomly search over design space
            new_experiment_inds = brnn.fast_search(main_data_scaled, objective, scaler, n_test, P=main_data)   ####TODO: ensure this works find without y-value in main_data_scaled
            new_experiments = main_data[new_experiment_inds]
            
            # remove training samples from main dataset  ###TODO: change indexing maybe
            #main_data = main_data[~new_experiment_inds]

            new_experiments = objective_time_collect(new_experiments)

            # compute objectives   
            target_found = []
            x_found = []
            ind_vars = []

            for sample in new_experiments:
                #Appending the objective value
                target_found.append(objective(sample))
                #Appending the 2 depedent variables
                x_found.append(sample[0,1:])

            target_found = np.array(target_found)
            x_found = np.array(x_found)
        
            # store the best objective found (so far)
            if (np.max(objective_found) > np.max(target_found)):
                x_objective_found.append(x_objective_found[np.argmax(objective_found)])
            else:
                x_objective_found.append(x_found[np.argmax(target_found)])
            
            objective_found.append(np.max([np.max(objective_found), np.max(target_found)]))
            '''
            print('found\n' + str(x_found))
            print('x_obj\n' + str(x_objective_found))
            print('y_found\n' + str(target_found))
            print('obj\n' + str(objective_found))
            '''
            #print(str(np.array(x_objective_found)[:,0]) + ' & ' + str(np.array(x_objective_found)[:,1]))
            '''
            plt.scatter(np.array(x_objective_found)[:,0],np.array(x_objective_found)[:,1])
            plt.show()
            '''
  
            train_data = np.vstack([train_data,new_experiments])
            time_array[trial,dtl] = time.time() - starting_time
            starting_time = time.time()

        ### fit model one last time to assess final prediction performance ### 
        # scale train and design space data
           
        if trial == 0:
            y_all = objective_found
            x_all = np.array(x_objective_found).reshape(1,-1)
        else:
            y_all = np.vstack([y_all, objective_found])
            x_all = np.vstack([x_all, np.array(x_objective_found).reshape(1,-1)])
        
        scaler = ZeroMaxScaler().fit(train_data)
        train_data_scaled = scaler.transform(train_data)
        main_data_scaled = scaler.transform(main_data)

        # fit model 
        N = len(train_data) * (t_eval - 1)
        brnn = miRNN(n_species=n_s, n_metabolites=0, n_controls=len(controls), n_hidden=16, N=N)
        brnn.fit(train_data_scaled)
        
        # assess prediction performance of end-point product
        pred, stdv, cov = brnn.predict(main_data_scaled)
        pred = scaler.inverse_transform(pred)
        stdv = scaler.inverse_transform(stdv)
        
        rvalue = linregress(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1)).rvalue
        sse_value = sse(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1))
        plt.scatter(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1), label="R = {:.3f}\nSSE = {:.3f}".format(rvalue, sse_value))
        
        plt.legend()
        #plt.show()
        objective_rval.append(rvalue)
        objective_sse.append(sse_value)
        
        '''
        plt.scatter(np.array(x_objective_found)[:,0],np.array(x_objective_found)[:,1])
        plt.show()
        '''
        '''
        # save data to dataframe
        dtl_df_i = pd.DataFrame()
        dtl_df_i['Trial'] = [trial]
        for j,obj_found in enumerate(objective_found):
            dtl_df_i[f'DTL {j}'] = [obj_found]
        dtl_df = pd.concat((dtl_df, dtl_df_i))
        
        # save data to dataframe
        dtl_df_r = pd.DataFrame()
        dtl_df_r['Trial'] = [trial]
        for j,r_val in enumerate(objective_rval):
            dtl_df_r[f'DTL {j}'] = [r_val]
        dtl_df_R = pd.concat((dtl_df_R, dtl_df_r))
        
        # save data to dataframe
        dtl_df_e = pd.DataFrame()
        dtl_df_e['Trial'] = [trial]
        for j,e in enumerate(objective_sse):
            dtl_df_e[f'DTL {j}'] = [e]
        dtl_df_sse = pd.concat((dtl_df_sse, dtl_df_e))
        
        dtl_df.to_csv("Fig3/miRNN_explore_exploit_FAST.csv", index=False)
        dtl_df_R.to_csv("Fig3/miRNN_explore_exploit_FAST_rvals.csv", index=False)
        dtl_df_sse.to_csv("Fig3/miRNN_explore_exploit_FAST_sse.csv", index=False)
        '''
    time_results.append(time.time())
    print(train_data)
    print(x_all)
    print(len(x_all))
    print(y_all)
    print(len(y_all))

    all = pd.DataFrame(np.hstack([y_all,x_all]))
    all.to_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/4-D/RNN/4D_" + str(n_init) +"_intial_" + str(n_test) + "_samp_"+str(n_dtl)+"_rounds_"+str(n_trials)+"_tanh.csv",index = False)

    ti = pd.DataFrame(time_results)
    ti.to_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/4-D/RNN/4D_time_" + str(n_init) +"_intial_" + str(n_test) + "_samp_"+str(n_dtl)+"_rounds_"+str(n_trials)+"_tanh.csv",index = False)

    r = pd.DataFrame(np.hstack([objective_rval,objective_sse]))
    r.to_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/4-D/RNN/4D_Rvals_" + str(n_init) +"_intial_" + str(n_test) + "_samp_"+str(n_dtl)+"_rounds_"+str(n_trials)+"_tanh.csv",index = False)
    print('asdfasdf')

def Explore_Exploit_3D():
    start0 = time.time()
    time_results = []
    time_results.append(start0)
    
    ####Initializes Grid Creation
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

    kernel = []
    exp_w = []
    t_eval = 11

    BO_obj = myBO.BO(ub,lb,step,dim = dim,exp_w = exp_w,kernel = kernel,system = PM_3.BO_intializer,bounds = bnds)
    org_t = time.time()
    BO_obj.auto_create_axis()
    BO_obj.iterate(0)

    main_conditions = BO_obj.descale(BO_obj.grid_vertices)
    row_len = len(main_conditions)
    params = {'legend.fontsize': 18,
          'figure.figsize': (8, 6),
         'axes.labelsize': 24,
         'axes.titlesize':24,
         'axes.linewidth':5,
         'xtick.labelsize':20,
         'ytick.labelsize':20}
    plt.rcParams.update(params)

    plt.rcParams['pdf.fonttype'] = 42
    
    from numpy.random import default_rng
    rng = default_rng(seed =  543)

    # number of trials 
    n_trials = 10

    # number of dtl cycles 
    n_dtl  = 5

    # define number of initial samples to train on
    n_init = 10

    # number of samples for next experiment 
    n_test = 40

    # number of species in model
    n_s = 1

    # number of resources
    n_r = 0

    # define all system variables 
    species = ['Biomass']
    outputs = []
    sys_var = species + outputs

    # define parameters in the objective function
    obj_params = []

    # define subset of controls
    controls = ['Light Intensity','Nitrogen','Quota']
    system_variables = species + outputs + controls

    # define an objective function (product of endpoint volume * endpoint product concentration)
    # where product concentration is the last column of the predicted output
    
    def objective(data, obj_params = 0): ###TODO This will need to be edited to call function, maybe
        return data[-1,0]
        #return trapezoid(y[:,0],time_interval)
    
    def objective_time_collect(data):
        '''
        Fills in the train_data with biomass over time
        '''
        obj = data[:,-1,1:]

        for row in range(len(obj)):
            y = PM_3.it(obj[row,:])[0,:]
            data[row,:,0] = y

        return data

    def main_expansion(data):
        '''
        This expands a matrix of coordinates to have the time dimension. This is needed for the RNN 
        '''
        num_rows,num_cols = data.shape
        n = 11
        zeros = np.zeros((num_rows, n, num_cols))
        # Stack the original matrix and the zeros along the new axis
        new_matrix = np.zeros((num_rows, 11, num_cols + 1))

        # Fill the new 3D matrix
        for i in range(num_rows):
            #Fill in conditions across time
            new_matrix[i,:,1:] = np.tile(data[i, :], (11, 1))
            #Give initial Biomass start
            new_matrix[i,0,0] = 0.0555

        return new_matrix

    main_data = main_expansion(main_conditions)

    # import data ## TODO Delete this and replace with list of grid arguments
    # determine random sets of initial experiments
    initial_exps = [rng.choice(row_len, n_init, replace=False) for _ in range(n_trials)]

    compare_ind =  rng.choice(row_len, 10000, replace=False)
    
    main_data[compare_ind,:,:] = objective_time_collect(main_data[compare_ind,:,:])
    # function to compute sum of squares error 
    def sse(a, b):
        return np.sum((a-b)**2)

    # init dataframe that stores DTL information
    dtl_df = pd.DataFrame()
    dtl_df_R = pd.DataFrame()
    dtl_df_sse = pd.DataFrame()
    elapsed_time = []

    x_all = []
    y_all = []
    time_array = np.zeros((n_trials,n_dtl))
    
    objective_rval = []
    objective_sse  = []
    
    for trial in range(n_trials):
        time_results.append(time.time())
        print('trial' + str(trial+1))
        #if trial == 1:
        #    continue

        # choose random set of training samples ##TODO edit to make avaliable for a Numpy matirx
        train_data = objective_time_collect(main_data[initial_exps[trial]])

        # keep track of objective 
        objective_found = []
        x_objective_found = []

        
        #train_data, train_obj_params, new_experiments, N = format_data(train_df, species, outputs, controls, obj_params=obj_params)

        # remove training samples from main dataset so that they're not selected more than once ##TODO edit to make avaliable for a Numpy matirx
        '''
        mask = np.ones(main_data.shape[0], dtype=bool)
        mask[initial_exps[trial]] = False
        #main_data = main_data[mask]
        print('1.5')
        m = []
        for batch_inds in np.array_split(np.arange(len(mask)), len(mask)//512):
            m.append(main_data[batch_inds])
        main_data = m
        '''

        # compute objectives ###TODO May need to edit for-loop
        target_found = []
        x_found = []

        for sample in train_data:
            #Appending the objective value
            target_found.append(objective(sample))
            #Appending the 2 depedent variables
            x_found.append(sample[0,1:])

        target_found = np.array(target_found)
        x_found = np.array(x_found)
        objective_found.append(np.max(target_found))
        x_objective_found.append(x_found[np.argmax(target_found)])
        
        # Search over full factorial and update model
        starting_time = time.time()
        for dtl in range(n_dtl):
            print(f"Running trial {trial+1}, cycle {dtl+1}")

            scaler = ZeroMaxScaler().fit(train_data)
            train_data_scaled = scaler.transform(train_data)
            main_data_scaled = scaler.transform(main_data)

            # fit model 
            N = len(train_data) * (t_eval - 1)
            brnn = miRNN(n_species=n_s, n_metabolites=0, n_controls=len(controls), n_hidden=16, N=N)
            brnn.fit(train_data_scaled,trial = trial)  ###Check Train_data format

            # assess prediction performance of end-point product
            pred, stdv, cov = brnn.predict(main_data_scaled)  ##Check main_data format
            pred = scaler.inverse_transform(pred)
            stdv = scaler.inverse_transform(stdv)
            
            rvalue = linregress(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1)).rvalue                                                                                                                                                                   
            sse_value = sse(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1))
            plt.scatter(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1), label="R = {:.3f}\nSSE = {:.3f}".format(rvalue, sse_value))
            plt.legend()
            #plt.show()
            
            objective_rval.append(rvalue)
            objective_sse.append(sse_value)
            '''
            plt.scatter(np.array(main_data[:, -1, 0]), pred[:,-1].reshape(1,-1), label="R = {:.3f}\nSSE = {:.3f}".format(rvalue, sse_value))
            plt.legend()
            plt.show()
            
            #Prove initial training set can identify correct trend
            fig, axs = plt.subplots(2,3)
            step = int((len(main_data)-1)/6)
            
            
            # This displays time series data uniform across the light intensities
            
            for i in range(6):
                rand = step*(i+1)

                axs[int(5/3-i/3),i%3].scatter(time_interval,main_data[rand,:,0],label = 'Data: ' + str(main_data[rand,0,-1]))
                axs[int(5/3-i/3),i%3].plot(time_interval,pred[rand,:,-1],label = 'Predicted')
                axs[int(5/3-i/3),i%3].legend()
            
            fig.text(0.08, 0.5, 'Biomass Concentration', va='center', rotation='vertical',fontsize='20')
            fig.text(.5, .04, 'Time', va='center',fontsize = '20')
            plt.suptitle('Time Series @ Different Light Intensities: Sigmoid',fontsize = '28')
            plt.show()
            
            plt.scatter(main_data[:,-1,1],main_data[:,-1,0], label = 'Actual')
            plt.scatter(main_data[:,-1,1],pred[:,-1,-1], label = 'Predicted')
            
            plt.title('Final Biomass Concentration across all light intensities, Sigmoid',fontsize = '14')
            plt.ylabel('Final Biomass Concentration',fontsize = '14')
            plt.xlabel('Light Intensity',fontsize = '14')
            plt.legend()
            plt.show()
            '''

            # randomly search over design space
            new_experiment_inds = brnn.fast_search(main_data_scaled, objective, scaler, n_test, P=main_data)   ####TODO: ensure this works find without y-value in main_data_scaled
            new_experiments = main_data[new_experiment_inds]
            
            # remove training samples from main dataset  ###TODO: change indexing maybe
            #main_data = main_data[~new_experiment_inds]

            new_experiments = objective_time_collect(new_experiments)

            # compute objectives   
            target_found = []
            x_found = []
            ind_vars = []

            for sample in new_experiments:
                #Appending the objective value
                target_found.append(objective(sample))
                #Appending the 2 depedent variables
                x_found.append(sample[0,1:])

            target_found = np.array(target_found)
            x_found = np.array(x_found)
        
            # store the best objective found (so far)
            if (np.max(objective_found) > np.max(target_found)):
                x_objective_found.append(x_objective_found[np.argmax(objective_found)])
            else:
                x_objective_found.append(x_found[np.argmax(target_found)])
            
            objective_found.append(np.max([np.max(objective_found), np.max(target_found)]))
            '''
            print('found\n' + str(x_found))
            print('x_obj\n' + str(x_objective_found))
            print('y_found\n' + str(target_found))
            print('obj\n' + str(objective_found))
            '''
            #print(str(np.array(x_objective_found)[:,0]) + ' & ' + str(np.array(x_objective_found)[:,1]))
            '''
            plt.scatter(np.array(x_objective_found)[:,0],np.array(x_objective_found)[:,1])
            plt.show()
            '''

            train_data = np.vstack([train_data,new_experiments])
            time_array[trial,dtl] = time.time() - starting_time
            starting_time = time.time()

        ### fit model one last time to assess final prediction performance ### 
        # scale train and design space data
           
        if trial == 0:
            y_all = objective_found
            x_all = np.array(x_objective_found).reshape(1,-1)
        else:
            y_all = np.vstack([y_all, objective_found])
            x_all = np.vstack([x_all, np.array(x_objective_found).reshape(1,-1)])
        
        scaler = ZeroMaxScaler().fit(train_data)
        train_data_scaled = scaler.transform(train_data)
        main_data_scaled = scaler.transform(main_data)

        # fit model 
        N = len(train_data) * (t_eval - 1)
        brnn = miRNN(n_species=n_s, n_metabolites=0, n_controls=len(controls), n_hidden=16, N=N)
        brnn.fit(train_data_scaled)
        
        # assess prediction performance of end-point product
        pred, stdv, cov = brnn.predict(main_data_scaled)
        pred = scaler.inverse_transform(pred)
        stdv = scaler.inverse_transform(stdv)
        
        rvalue = linregress(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1)).rvalue
        sse_value = sse(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1))
        plt.scatter(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1), label="R = {:.3f}\nSSE = {:.3f}".format(rvalue, sse_value))
        
        plt.legend()
        #plt.show()
        objective_rval.append(rvalue)
        objective_sse.append(sse_value)

        '''
        plt.scatter(np.array(x_objective_found)[:,0],np.array(x_objective_found)[:,1])
        plt.show()
        '''
        '''
        # save data to dataframe
        dtl_df_i = pd.DataFrame()
        dtl_df_i['Trial'] = [trial]
        for j,obj_found in enumerate(objective_found):
            dtl_df_i[f'DTL {j}'] = [obj_found]
        dtl_df = pd.concat((dtl_df, dtl_df_i))
        
        # save data to dataframe
        dtl_df_r = pd.DataFrame()
        dtl_df_r['Trial'] = [trial]
        for j,r_val in enumerate(objective_rval):
            dtl_df_r[f'DTL {j}'] = [r_val]
        dtl_df_R = pd.concat((dtl_df_R, dtl_df_r))
        
        # save data to dataframe
        dtl_df_e = pd.DataFrame()
        dtl_df_e['Trial'] = [trial]
        for j,e in enumerate(objective_sse):
            dtl_df_e[f'DTL {j}'] = [e]
        dtl_df_sse = pd.concat((dtl_df_sse, dtl_df_e))
        
        dtl_df.to_csv("Fig3/miRNN_explore_exploit_FAST.csv", index=False)
        dtl_df_R.to_csv("Fig3/miRNN_explore_exploit_FAST_rvals.csv", index=False)
        dtl_df_sse.to_csv("Fig3/miRNN_explore_exploit_FAST_sse.csv", index=False)
        '''

    time_results.append(time.time())
    print(train_data)
    print(x_all)
    print(len(x_all))
    print(y_all)
    print(len(y_all))

    all = pd.DataFrame(np.hstack([y_all,x_all]))
    all.to_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/3-D/RNN/4D_" + str(n_init) +"_intial_" + str(n_test) + "_samp_"+str(n_dtl)+"_rounds_"+str(n_trials)+"_tanh.csv",index = False)

    ti = pd.DataFrame(time_results)
    ti.to_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/3-D/RNN/4D_time_" + str(n_init) +"_intial_" + str(n_test) + "_samp_"+str(n_dtl)+"_rounds_"+str(n_trials)+"_tanh.csv",index = False)

    r = pd.DataFrame(np.hstack([objective_rval,objective_sse]))
    r.to_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/3-D/RNN/4D_Rvals_" + str(n_init) +"_intial_" + str(n_test) + "_samp_"+str(n_dtl)+"_rounds_"+str(n_trials)+"_tanh.csv",index = False)
    print('asdfasdf')

def Explore_Exploit_2D():
    start0 = time.time()
    time_results = []
    time_results.append(start0)
    
    ####Initializes Grid Creation
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

    kernel = []
    exp_w = []
    t_eval = 11

    BO_obj = myBO.BO(ub,lb,step,dim = dim,exp_w = exp_w,kernel = kernel,system = PM_2.BO_intializer,bounds = bnds)
    org_t = time.time()
    BO_obj.auto_create_axis()
    BO_obj.iterate(0)

    main_conditions = BO_obj.descale(BO_obj.grid_vertices)
    row_len = len(main_conditions)
    params = {'legend.fontsize': 18,
          'figure.figsize': (8, 6),
         'axes.labelsize': 24,
         'axes.titlesize':24,
         'axes.linewidth':5,
         'xtick.labelsize':20,
         'ytick.labelsize':20}
    plt.rcParams.update(params)

    plt.rcParams['pdf.fonttype'] = 42
    
    from numpy.random import default_rng
    rng = default_rng(seed =  543)

    # number of trials 
    n_trials = 10

    # number of dtl cycles 
    n_dtl  = 20

    # define number of initial samples to train on
    n_init = 4

    # number of samples for next experiment 
    n_test = 2

    # number of species in model
    n_s = 1

    # number of resources
    n_r = 0

    # define all system variables 
    species = ['Biomass']
    outputs = []
    sys_var = species + outputs

    # define parameters in the objective function
    obj_params = []

    # define subset of controls
    controls = ['Light Intensity','Nitrogen']
    system_variables = species + outputs + controls

    # define an objective function (product of endpoint volume * endpoint product concentration)
    # where product concentration is the last column of the predicted output
    
    def objective(data, obj_params = 0): ###TODO This will need to be edited to call function, maybe
        return data[-1,0]
        #return trapezoid(y[:,0],time_interval)
    
    def objective_time_collect(data):
        '''
        Fills in the train_data with biomass over time
        '''
        obj = data[:,-1,1:]

        for row in range(len(obj)):
            y = PM_2.it(obj[row,:])[0,:]
            data[row,:,0] = y

        return data

    def main_expansion(data):
        '''
        This expands a matrix of coordinates to have the time dimension. This is needed for the RNN 
        '''
        num_rows,num_cols = data.shape
        n = 11
        zeros = np.zeros((num_rows, n, num_cols))
        # Stack the original matrix and the zeros along the new axis
        new_matrix = np.zeros((num_rows, 11, num_cols + 1))

        # Fill the new 3D matrix
        for i in range(num_rows):
            #Fill in conditions across time
            new_matrix[i,:,1:] = np.tile(data[i, :], (11, 1))
            #Give initial Biomass start
            new_matrix[i,0,0] = 0.0555

        return new_matrix

    main_data = main_expansion(main_conditions)

    # import data ## TODO Delete this and replace with list of grid arguments
    # determine random sets of initial experiments
    initial_exps = [rng.choice(row_len, n_init, replace=False) for _ in range(n_trials)]
    

    compare_ind =  rng.choice(row_len, 10000, replace=False)
    
    main_data[compare_ind,:,:] = objective_time_collect(main_data[compare_ind,:,:])
    # function to compute sum of squares error 
    def sse(a, b):
        return np.sum((a-b)**2)

    # init dataframe that stores DTL information
    dtl_df = pd.DataFrame()
    dtl_df_R = pd.DataFrame()
    dtl_df_sse = pd.DataFrame()
    elapsed_time = []

    x_all = []
    y_all = []
    time_array = np.zeros((n_trials,n_dtl))
    
    objective_rval = []
    objective_sse  = []
    
    for trial in range(n_trials):
        time_results.append(time.time())
        print('trial' + str(trial+1))
        #if trial == 1:
        #    continue

        # choose random set of training samples ##TODO edit to make avaliable for a Numpy matirx
        train_data = objective_time_collect(main_data[initial_exps[trial]])

        # keep track of objective 
        objective_found = []
        x_objective_found = []

        
        #train_data, train_obj_params, new_experiments, N = format_data(train_df, species, outputs, controls, obj_params=obj_params)

        # remove training samples from main dataset so that they're not selected more than once ##TODO edit to make avaliable for a Numpy matirx
        '''
        mask = np.ones(main_data.shape[0], dtype=bool)
        mask[initial_exps[trial]] = False
        #main_data = main_data[mask]
        print('1.5')
        m = []
        for batch_inds in np.array_split(np.arange(len(mask)), len(mask)//512):
            m.append(main_data[batch_inds])
        main_data = m
        '''

        # compute objectives ###TODO May need to edit for-loop
        target_found = []
        x_found = []

        for sample in train_data:
            #Appending the objective value
            target_found.append(objective(sample))
            #Appending the 2 depedent variables
            x_found.append(sample[0,1:])

        target_found = np.array(target_found)
        x_found = np.array(x_found)
        objective_found.append(np.max(target_found))
        x_objective_found.append(x_found[np.argmax(target_found)])
        
        # Search over full factorial and update model
        starting_time = time.time()
        for dtl in range(n_dtl):
            print(f"Running trial {trial+1}, cycle {dtl+1}")

            scaler = ZeroMaxScaler().fit(train_data)
            train_data_scaled = scaler.transform(train_data)
            main_data_scaled = scaler.transform(main_data)

            # fit model 
            N = len(train_data) * (t_eval - 1)
            brnn = miRNN(n_species=n_s, n_metabolites=0, n_controls=len(controls), n_hidden=16, N=N)
            brnn.fit(train_data_scaled,trial = trial)  ###Check Train_data format

            # assess prediction performance of end-point product
            pred, stdv, cov = brnn.predict(main_data_scaled)  ##Check main_data format
            pred = scaler.inverse_transform(pred)
            stdv = scaler.inverse_transform(stdv)
            
            rvalue = linregress(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1)).rvalue                                                                                                                                                                   
            sse_value = sse(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1))
            plt.scatter(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1), label="R = {:.3f}\nSSE = {:.3f}".format(rvalue, sse_value))
            plt.legend()
            #plt.show()
            
            objective_rval.append(rvalue)
            objective_sse.append(sse_value)
            '''
            plt.scatter(np.array(main_data[:, -1, 0]), pred[:,-1].reshape(1,-1), label="R = {:.3f}\nSSE = {:.3f}".format(rvalue, sse_value))
            plt.legend()
            plt.show()
            
            #Prove initial training set can identify correct trend
            fig, axs = plt.subplots(2,3)
            step = int((len(main_data)-1)/6)
            
            
            # This displays time series data uniform across the light intensities
            
            for i in range(6):
                rand = step*(i+1)

                axs[int(5/3-i/3),i%3].scatter(time_interval,main_data[rand,:,0],label = 'Data: ' + str(main_data[rand,0,-1]))
                axs[int(5/3-i/3),i%3].plot(time_interval,pred[rand,:,-1],label = 'Predicted')
                axs[int(5/3-i/3),i%3].legend()
            
            fig.text(0.08, 0.5, 'Biomass Concentration', va='center', rotation='vertical',fontsize='20')
            fig.text(.5, .04, 'Time', va='center',fontsize = '20')
            plt.suptitle('Time Series @ Different Light Intensities: Sigmoid',fontsize = '28')
            plt.show()
            
            plt.scatter(main_data[:,-1,1],main_data[:,-1,0], label = 'Actual')
            plt.scatter(main_data[:,-1,1],pred[:,-1,-1], label = 'Predicted')
            
            plt.title('Final Biomass Concentration across all light intensities, Sigmoid',fontsize = '14')
            plt.ylabel('Final Biomass Concentration',fontsize = '14')
            plt.xlabel('Light Intensity',fontsize = '14')
            plt.legend()
            plt.show()
            '''

            # randomly search over design space
            new_experiment_inds = brnn.fast_search(main_data_scaled, objective, scaler, n_test, P=main_data)   ####TODO: ensure this works find without y-value in main_data_scaled
            new_experiments = main_data[new_experiment_inds]
            
            # remove training samples from main dataset  ###TODO: change indexing maybe
            #main_data = main_data[~new_experiment_inds]

            new_experiments = objective_time_collect(new_experiments)

            # compute objectives   
            target_found = []
            x_found = []
            ind_vars = []

            for sample in new_experiments:
                #Appending the objective value
                target_found.append(objective(sample))
                #Appending the 2 depedent variables
                x_found.append(sample[0,1:])

            target_found = np.array(target_found)
            x_found = np.array(x_found)
        
            # store the best objective found (so far)
            if (np.max(objective_found) > np.max(target_found)):
                x_objective_found.append(x_objective_found[np.argmax(objective_found)])
            else:
                x_objective_found.append(x_found[np.argmax(target_found)])
            
            objective_found.append(np.max([np.max(objective_found), np.max(target_found)]))
            '''
            print('found\n' + str(x_found))
            print('x_obj\n' + str(x_objective_found))
            print('y_found\n' + str(target_found))
            print('obj\n' + str(objective_found))
            '''
            #print(str(np.array(x_objective_found)[:,0]) + ' & ' + str(np.array(x_objective_found)[:,1]))
            '''
            plt.scatter(np.array(x_objective_found)[:,0],np.array(x_objective_found)[:,1])
            plt.show()
            '''

            train_data = np.vstack([train_data,new_experiments])
            time_array[trial,dtl] = time.time() - starting_time
            starting_time = time.time()

        ### fit model one last time to assess final prediction performance ### 
        # scale train and design space data
           
        if trial == 0:
            y_all = objective_found
            x_all = np.array(x_objective_found).reshape(1,-1)
        else:
            y_all = np.vstack([y_all, objective_found])
            x_all = np.vstack([x_all, np.array(x_objective_found).reshape(1,-1)])
        
        scaler = ZeroMaxScaler().fit(train_data)
        train_data_scaled = scaler.transform(train_data)
        main_data_scaled = scaler.transform(main_data)

        # fit model 
        N = len(train_data) * (t_eval - 1)
        brnn = miRNN(n_species=n_s, n_metabolites=0, n_controls=len(controls), n_hidden=16, N=N)
        brnn.fit(train_data_scaled)
        
        # assess prediction performance of end-point product
        pred, stdv, cov = brnn.predict(main_data_scaled)
        pred = scaler.inverse_transform(pred)
        stdv = scaler.inverse_transform(stdv)
        
        rvalue = linregress(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1)).rvalue
        sse_value = sse(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1))
        plt.scatter(np.array(main_data[compare_ind, -1, 0]), pred[compare_ind,-1].reshape(1,-1), label="R = {:.3f}\nSSE = {:.3f}".format(rvalue, sse_value))
        
        plt.legend()
        #plt.show()
        objective_rval.append(rvalue)
        objective_sse.append(sse_value)

        '''
        plt.scatter(np.array(x_objective_found)[:,0],np.array(x_objective_found)[:,1])
        plt.show()
        '''
        '''
        # save data to dataframe
        dtl_df_i = pd.DataFrame()
        dtl_df_i['Trial'] = [trial]
        for j,obj_found in enumerate(objective_found):
            dtl_df_i[f'DTL {j}'] = [obj_found]
        dtl_df = pd.concat((dtl_df, dtl_df_i))
        
        # save data to dataframe
        dtl_df_r = pd.DataFrame()
        dtl_df_r['Trial'] = [trial]
        for j,r_val in enumerate(objective_rval):
            dtl_df_r[f'DTL {j}'] = [r_val]
        dtl_df_R = pd.concat((dtl_df_R, dtl_df_r))
        
        # save data to dataframe
        dtl_df_e = pd.DataFrame()
        dtl_df_e['Trial'] = [trial]
        for j,e in enumerate(objective_sse):
            dtl_df_e[f'DTL {j}'] = [e]
        dtl_df_sse = pd.concat((dtl_df_sse, dtl_df_e))
        
        dtl_df.to_csv("Fig3/miRNN_explore_exploit_FAST.csv", index=False)
        dtl_df_R.to_csv("Fig3/miRNN_explore_exploit_FAST_rvals.csv", index=False)
        dtl_df_sse.to_csv("Fig3/miRNN_explore_exploit_FAST_sse.csv", index=False)
        '''

    time_results.append(time.time())
    print(train_data)
    print(x_all)
    print(len(x_all))
    print(y_all)
    print(len(y_all))

    all = pd.DataFrame(np.hstack([y_all,x_all]))
    all.to_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/2-D/RNN/2D_" + str(n_init) +"_intial_" + str(n_test) + "_samp_"+str(n_dtl)+"_rounds_"+str(n_trials)+"_tanh.csv",index = False)

    ti = pd.DataFrame(time_results)
    ti.to_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/2-D/RNN/2D_time_" + str(n_init) +"_intial_" + str(n_test) + "_samp_"+str(n_dtl)+"_rounds_"+str(n_trials)+"_tanh.csv",index = False)

    r = pd.DataFrame(np.hstack([objective_rval,objective_sse]))
    r.to_csv("/Users/samuelhower/Documents/Classes/Research/Zavala/Summer of 2023/Results/Comparing P to non P/2-D/RNN/2D_Rvals_" + str(n_init) +"_intial_" + str(n_test) + "_samp_"+str(n_dtl)+"_rounds_"+str(n_trials)+"_tanh.csv",index = False)
    print('asdfasdf')

Explore_Exploit_5D()

            
