### Global imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

SHOW_PLOTS = False #Please Change to True if you want the windows to pop up

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
})

## Dictionaries and Variables

#Prey and predator values
Part1_Params = {
    "alpha": 1.0,   #Rabbit birth rate per day
    "beta":  0.1,   #Predation rate per day
    "delta": 0.02,  #Fox birth rate per day
    "gamma": 0.5    #Fox death rate per day
}

#Zombie outbreak values
Part2_Params = {
    "alpha":   0.0,    #Human birth rate per day
    "beta":    1e-4,   #Natural human death rate per day
    "gamma":   9.5e-3, #Zombification rate per day
    "delta":   1e-4,   #Zombie destruction rate per day
    "epsilon": 1e-4,   #Resurrection rate
}

STEP    = 0.01  #Step length (days)
TF_P1   = 100.0 #The end running time for part one
TF_P2   = 5.0   #The end running time for part two

TF_CONV = 20.0 #The end time for the convergence 
H_EULER = [0.04, 0.02, 0.01, 0.005, 0.0025] #Euler step sizes (days)
H_RK4   = [0.4, 0.2, 0.1, 0.05, 0.025]      #RK4 step sizes (days) chosen to be larger as RK4 is much more accurate

## Functions

def finish_figure(title):
    ''' Saves the current figure to the working directory, 
        and if SHOW_PLOTS is set to True, will display the plot.
    '''
    plt.savefig(f'./{title}.png', dpi=300, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

def outbreaks_model(gamma, a):
    return a / gamma

def outbreaks_model_p(gamma, a, p):
    '''Running T = a / gamma^p. p is close to 1.'''
    return a / gamma**p


def find_crossover_times(populations, times):
    '''Returns the time when H first falls below Z, or None. Utilises linear interpolation'''

    H, Z = populations[0], populations[1]
    below = H < Z 

    if not below.any(): 
        return None #If the humans always outnumber the zombies

    crossover = np.argmax(below) #Firsst index where H < Z

    gap0 = H[crossover - 1] - Z[crossover - 1] #As it will positive just before
    gap1 = H[crossover] - Z[crossover] #As it will be negative just after/on it
    fraction = gap0 / (gap0 - gap1) #The fraction of the step to zero
    crossover_time = times[crossover - 1] + (fraction * (times[crossover] - times[crossover - 1])) #The grid level accurate time of when the crossover occurs
    

    return crossover_time
    
def conserved_quantity(populations, parameters):

    x, y = populations[0], populations[1]

    alpha = parameters["alpha"]
    beta  = parameters["beta"]
    delta = parameters["delta"]
    gamma = parameters["gamma"]

    return (delta * x) + (beta * y) - (gamma * np.log(x)) - (alpha * np.log(y)) #This is the V

def relative_drift(populations, parameters):

    conserved_quant = conserved_quantity(populations, parameters)
    V0 = conserved_quant[0]

    return np.abs(conserved_quant - V0) / np.abs(V0) #This is the drift error

def max_drift_for_step(method, step_sizes, tf):
    '''The relative maximum drift for V (the conserved quantity), using Euler or RK4 method'''

    max_drifts = []
    for h in step_sizes:
        times, populations = method(derivative_rabbit_fox, 0.0, tf, h, Part1_Params, state0_rabbit_fox) #Changes the step size h
        max_drift = max(relative_drift(populations, Part1_Params))
        max_drifts.append(max_drift)

    return max_drifts


## Plotting Functions

def plot_rabbit_fox(title, times, populations, parameters):

    x_stat = parameters["alpha"]/parameters["beta"] #Nullcline for rabbits
    y_stat = parameters["gamma"]/parameters["delta"] #Nullcline for foxes

    x_peaks, x_properties = find_peaks(populations[0]) #Returns the array of peak indices and a dictionary of their properties
    t_peaks = times[x_peaks] #Array of times for when the rabbits peak
    periods = np.diff(t_peaks) #There is a period for the gap between neighbouring peaks
    print("T = %.3f, spread = %.3f" % (periods.mean(), periods.std())) #Period, mean and spread (standard deviation) over all cycles

    start, end = x_peaks[0], x_peaks[1] #Start and end of a period
    print("Start point:", (populations[0][start], populations[1][start]))
    print("End point:  ", (populations[0][end], populations[1][end]))
    start_point = (populations[0][start], populations[1][start])
    end_point = ((populations[0][end], populations[1][end]))

    # Using matplotlib.pyplot.subplots to produce a clean output
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot populations varying with time on the same graph
    ax[0].plot(times, populations[0], 'b-', label='Rabbits')
    ax[0].plot(times, populations[1], 'r-', label='Foxes')
    ax[0].set_title('Populations')
    ax[0].set_xlabel('Time $t$ (days)')
    ax[0].set_ylabel('No. of animals')
    ax[0].legend(loc='upper right')
    ax[0].set_ylim(0, max(populations[0])+50) # Configuring the y-limits
    ax[0].grid(True) #Provides ease of interpreting

    # Plotting in Phase space
    ax[1].plot(populations[0], populations[1], 'b-')
    ax[1].axhline(y=0, color='grey') #Horizontal Nullcline
    ax[1].axvline(x=0, color='grey') #Vertical Nullcline 
    ax[1].axvline(y_stat, color='grey', linestyle='--', label=r'$dy/dt=0$') #Vertical non-0 Nullcline
    ax[1].axhline(x_stat, color='grey', linestyle=':',  label=r'$dx/dt=0$') #Horizontal non-0 Nullcline
    ax[1].plot(y_stat, x_stat, 'k*', label= 'Fixed point')
    ax[1].plot(start_point[0], start_point[1], 'go', label= 'Start point')
    ax[1].plot(end_point[0], end_point[1], 'r*', label='End point')
    ax[1].set_title('Trajectories')
    ax[1].set_xlabel('No. Rabbits')
    ax[1].set_ylabel('No. Foxes')
    ax[1].legend()
    plt.tight_layout() #Ajusts spacing to prevent overlapping
    finish_figure(title)

def plot_zombie(title, times, populations):
    # Using matplotlib.pyplot.subplots to produce a clean output
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Plot populations varying with time on the same graph
    ax.plot(times, populations[0], 'b-', label='Human')
    ax.plot(times, populations[1], 'g-', label='Zombie')
    ax.plot(times, populations[2], 'r-', label='Dead')

    crossover_time = find_crossover_times(populations, times)                                       
    if crossover_time is not None:
        ax.axvline(crossover_time, label = 'Crossover time $(H=Z)$') #Draws a line for the crossover time
        print(f'Crossover occurs at {crossover_time}')

    ax.set_title('Populations')
    ax.set_xlabel('Time $t$ (days)')
    ax.set_ylabel('Population Levels')
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(populations[0])+50) # Configuring the y-limits
    ax.set_xlim(0, times[-1])
    ax.grid(True) #Provides ease of interpreting

    plt.tight_layout() #Ajusts spacing to prevent overlapping
    finish_figure(title)

def plot_gamma_variation(title, times, runs, gammas):
    """ Plots the population levels against time as before, 
        however a colour gradient for gamma is introduced.
        There are 3 separate panels.
    """

    titles = ['Humans $H$', 'Zombies $Z$', 'Dead $D$'] #For the three graphs

    norm = colors.Normalize(vmin=gammas.min(), vmax=gammas.max()) #This transforms gamma into numbers from 0 through 1
    cmap = plt.get_cmap('plasma') #This transforms those 0 through to 1 numbers into a colour map

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharex = True, constrained_layout = True)

    for i, panel in enumerate(axes): #i is iterating through the three populations

        for j, k in enumerate(gammas):  #j is picking which run of the function to display
            panel.plot(times, runs[j, i], color = cmap(norm(k)))
        panel.set_title(titles[i])
        panel.set_xlabel('Time $t$ (days)')
        panel.grid(True)
    axes[0].set_ylabel('Population') #Puts the y axis solely on the first graph which is all that is appropriate

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap) #This sorts the colour bar
    fig.colorbar(sm, ax=axes, label=r'Zombification rate: $\gamma$ (day$^{-1}$)')

    finish_figure(title)


def plot_dz_by_dt(title, zombie_rates, times):
    '''Mapping the rate of change of zombies over time, 
    used to determine a reasonable start time for a zombie outbreak
    '''

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(times, zombie_rates, 'g.')
    ax.set_title('Rate of change of Zombie population over time')
    ax.set_xlabel(r'Time (days)')
    ax.set_ylabel(r'Zombie rate of change $\frac{dZ}{dt}$')

    ax.grid(True) #Provides ease of interpreting
    plt.tight_layout() #Ajusts spacing to prevent overlapping
    finish_figure(title)



def plot_outbreaks(title, gammas, outbreaks, model, popt, h):
    ''' Plots the outbreak time for the system for changing gammas.
        The top graph shows outbreak times with the fitted curve.
        The bottom graph shows the residuals with error bars at ±h/2
    '''

    fit_curve = model(gammas, *popt) #Fits a curve at every gamma
    residuals = outbreaks - fit_curve #How far it should be from the fit

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex= True, gridspec_kw={'height_ratios': [3,1]})


    ax[0].plot(gammas, outbreaks, 'r.')
    ax[0].plot(gammas, fit_curve, 'k-', label='Fit')
    ax[0].set_title('Outbreak Time Varying over Gamma Range')
    ax[0].set_ylabel('Outbreak time (days)')
    ax[0].legend()
    ax[0].grid(True) #Provides ease of interpreting

    ax[1].errorbar(gammas, residuals, yerr= h/2, fmt="r.", ecolor= 'grey', capsize= 2) #Fitting error bars
    ax[1].axhline(0, color= 'k', linewidth= 0.8) #Zero line, where we would find a perfect fit
    ax[1].set_xlabel(r'$\gamma (days^{-1})$')
    ax[1].set_ylabel('Residual (days)')
    ax[1].grid(True)

    plt.tight_layout() #Ajusts spacing to prevent overlapping
    finish_figure(title)

def plot_taus(title, taus, H_max, delta_max):
    '''Plots the final human population against tau,
       with delta maximum represented by a colour scale.
    '''

    tau_grid, delta_grid = np.meshgrid(taus, delta_max, indexing='ij') #creates two arrays containing all combos of tau and delta_max

    norm = colors.Normalize(vmin=delta_max.min(), vmax=delta_max.max())
    cmap = plt.get_cmap('plasma')

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

    #Using a scatter plot we want individual points with varying colour
    scatter = ax.scatter(tau_grid.flatten(), H_max.flatten(), c=delta_grid.flatten(), cmap=cmap, norm=norm)
    #tau_grid.flatten (x coords), H_max.flatten (y coords), delta_grid.flatten (colours)

    ax.set_title('Final Human Population Varying over Tau')
    ax.set_xlabel(r'$\tau$ (days)')
    ax.set_ylabel('Final Human Population')
    ax.grid(True)

    #The colour bar creates a colour scale for the values of delta_max
    fig.colorbar(scatter, ax=ax, label=r'Maximum destruction rate: $\delta_{\max}$ (day$^{-1}$)')

    finish_figure(title)

def plot_errors(title, times, drift_euler, drift_rk4, 
                hs_euler, errors_euler, fit_euler, 
                hs_rk4, errors_rk4, fit_rk4):
    ''' Plots two graphs. One of the drift against the time, 
        and the other the maximum drift against the step size.
        Utilises fit_euler and fit_rk4 as the slope and intercept pairs from numpy's polyfit
    '''
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))

    #Drift against time (dropps when drift is 0 so as not ot have a maths error)
    ax[0].semilogy(times[1:], drift_euler[1:], label = 'Euler') 
    ax[0].semilogy(times[1:], drift_rk4[1:], label = 'RK4') 

    #The points with no connecting line
    ax[1].loglog(hs_euler, errors_euler, 'o', color='C0', label='Euler')
    ax[1].loglog(hs_rk4,   errors_rk4,   's', color='C1', label='RK4')    

    h_euler = np.array(hs_euler) #Numpy array is more useful
    h_rk4 = np.array(hs_rk4)

    #Fit the straight lines using polyfit which produces log(err) = slope * log(h) + intercept
    ax[1].loglog(h_euler, np.exp(fit_euler[1]) * h_euler**fit_euler[0], '--', color='C0', label=f'Euler fit, slope = {fit_euler[0]:.2f}')
    ax[1].loglog(h_rk4, np.exp(fit_rk4[1]) * h_rk4**fit_rk4[0], '--', color='C1', label=f'RK4 fit, slope = {fit_rk4[0]:.2f}')

    ax[0].set_title(rf'Drift of conserved quantity $V$ against time ($h$ = {STEP} days)')
    ax[0].set_xlabel(r'Time $t$ (days)')
    ax[0].set_ylabel(r'Relative drift $\varepsilon$')
    ax[1].set_title(rf'Maximum drift against step size ($t_f$ = {TF_CONV:g} days)')
    ax[1].set_xlabel(r'Step size $h$ (days)')
    ax[1].set_ylabel(r'Maximum relative drift $\varepsilon_{\max}$')

    ax[0].legend()
    ax[0].grid(True, which='both')
    ax[1].legend()
    ax[1].grid(True, which='both')

    plt.tight_layout()
    finish_figure(title)


## Derivative functions
    
def derivative_rabbit_fox(t, parameters, state):
    ## A singular function for the rates of change, for use in Part 1

    x, y = state

    alpha = parameters["alpha"]
    beta  = parameters["beta"]
    delta = parameters["delta"]
    gamma = parameters["gamma"]

    r_rate = ((alpha * x) - (beta * x * y)) #Rate of change of prey
    f_rate = ((delta * x * y) - (gamma * y)) #Rate of change of predator
    return np.array([r_rate, f_rate])

def derivative_zombie(time, parameters, state):
    '''A singular function for the rates of change, for use in Parts 2 and 3'''
    
    h, z, d = state #For Human, Zombie, and Dead

    delta0      = parameters["delta"] #Initial zombie destruction rate
    delta_max   = parameters.get("delta_max", delta0) #Ceiling of zombie destruction rate. If not appended to parameters then remains delta0
    tau         = parameters.get("tau", 1) #Rate of adaptability of humans. If not appended to parameters then remains 1

    alpha   = parameters["alpha"]
    beta    = parameters["beta"]
    delta   = delta_max - (delta_max - delta0)*np.exp(-(time/tau))
    gamma   = parameters["gamma"]
    epsilon = parameters["epsilon"]

    h_rate = ((alpha * h) - (beta * h) - (gamma * h * z))
    z_rate = ((gamma * h * z) - (delta * h * z) + (epsilon * d))
    d_rate = ((beta * h) + (delta * h * z) - (epsilon * d))

    return np.array([h_rate, z_rate, d_rate])


def Euler(derivative_func, ti, tf, step, parameters, state0): 
    """ Performs the Euler method on the state provided (state0), using the parameters provided 
        (parameters), and the derivation function provided (derivative_func). Additionally:
        ti is the initial time in days, tf is the final time in days, h is the step size in days.
        Returns an array of equally spaced times with their corresponding population levels.
    """

    Nstep = round((tf - ti)/step) #Calculation of the number of steps
    t_array = np.linspace(ti, tf, Nstep+1) #Generating array of times

    Population = np.zeros((len(state0), len(t_array))) #Initialising a single state vector
    Population[:, 0] = state0 #Assigns initial population array

    ## Iteration implementation of Euler's method
    for i in range(0, len(t_array)-1):
 
        t = t_array[i]
        state = Population[:,i] #Current state vector

        Population[:, i+1] = state + step*derivative_func(t, parameters, state)

    return t_array, Population
    #Utilising this data in a graphing function makes this Euler function reusable

def RK4(derivative_func, ti, tf, step, parameters, state0):
    """ Performs the RK4 method on the state provided (state0), using the parameters provided 
        (parameters), and the derivation function provided (derivative_func). Additionally:
        ti is the initial time in days, tf is the final time in days, h is the step size in days.
        Returns an array of equally spaced times with their corresponding population levels.
    """
    

    Nstep = round((tf - ti)/step) #Calculation of the number of steps
    t_array = np.linspace(ti, tf, Nstep+1) #Generating array of times

    Population = np.zeros((len(state0), len(t_array))) #Initialising a single state vector
    Population[:, 0] = state0 #Assigns initial population array
   
    ## Iteration implementation of RK4 method
    for i in range(0, len(t_array)-1):
        t = t_array[i]          #Current time
        state = Population[:,i]    #Current state vector
        k1 = derivative_func(t,           parameters, state) 
        k2 = derivative_func(t + (step/2),   parameters, state + ((step/2)*k1))
        k3 = derivative_func(t + (step/2),   parameters, state + ((step/2)*k2))
        k4 = derivative_func(t + step,       parameters, state + (step*k3))
        
        Population[:, i+1] = state + step*(k1 + (2*k2) + (2*k3) + k4)/6

    return t_array, Population
    #Utilising this data in a graphing function makes this RK4 function reusable
        


state0_rabbit_fox = np.array([100, 20]) #Initial number of rabbits and foxes
state0_zombie = np.array([1000, 5, 0]) #Initial number of humans, zombies and dead

def run_Euler_animals():
    '''Euler method to plot rabbits and foxes'''

    times, populations = Euler(derivative_rabbit_fox, 0.0, TF_P1, STEP, Part1_Params, state0_rabbit_fox)
    plot_rabbit_fox('Euler_rabbit_fox', times, populations, Part1_Params)

def run_RK4_animals():
    '''RK4 method to plot rabbits and foxes'''

    times, populations = RK4(derivative_rabbit_fox, 0.0, TF_P1, STEP, Part1_Params, state0_rabbit_fox)
    plot_rabbit_fox('RK4_rabbit_fox', times, populations, Part1_Params)

def run_error_analysis():
    '''Calculates drift error for Euler and RK4 method of the animalas parameter set 1'''

    times_euler, populations_euler = Euler(derivative_rabbit_fox, 0.0, TF_P1, STEP, Part1_Params, state0_rabbit_fox)    
    times_RK4, populations_RK4 = RK4(derivative_rabbit_fox, 0.0, TF_P1, STEP, Part1_Params, state0_rabbit_fox)    
        
    drift_euler = relative_drift(populations_euler, Part1_Params)
    drift_RK4 = relative_drift(populations_RK4, Part1_Params)

    print(f'The maximum drift error for the Euler method is {drift_euler.max()}')
    print(f'The maximum drift error for the RK4 method is {drift_RK4.max()}')

    #Maximum drift for a range of step sizes and then a straight fit on log-log axes
    errs_euler = max_drift_for_step(Euler, H_EULER, TF_CONV)
    errs_RK4 = max_drift_for_step(RK4, H_RK4, TF_CONV)

    fit_euler = np.polyfit(np.log(H_EULER), np.log(errs_euler), 1) #contains slope and intercept
    fit_rk4 = np.polyfit(np.log(H_RK4), np.log(errs_RK4), 1)

    print(f'Euler convergence slope: {fit_euler[0]:.2f}')
    print(f'RK4 convergence slope: {fit_rk4[0]:.2f}')

    #RK4 uses 4 derivative evaluations per step therefore a step 4 times larger costs the same processing time as Euler
    times_cost, populations_cost = RK4(derivative_rabbit_fox, 0.0, TF_P1, 4*STEP, Part1_Params, state0_rabbit_fox)
    drift_cost = relative_drift(populations_cost, Part1_Params)
    print(f'Equal cost: Euler (h = {STEP}) max drift {drift_euler.max():.2e}, RK4 (h = {4*STEP}) max drift {drift_cost.max():.2e}')

    plot_errors('Error_analysis', times_euler, drift_euler, drift_RK4, H_EULER, errs_euler, fit_euler, H_RK4, errs_RK4, fit_rk4)

def run_RK4_zombies():
    '''RK4 method to plot the zombie outbreak'''

    times, populations = RK4(derivative_zombie, 0.0, TF_P2, STEP, Part2_Params, state0_zombie)
    plot_zombie('RK4_zombies', times, populations)

def run_gamma_variation(): 
    '''RK4 method for zombie outbreak with variations in gamma'''

    gammas = np.linspace(2e-3, 18e-3, 7) #Gammas equally spaced around a suitable range
    gammas = np.append(gammas, 9.5e-3) #Appending on the original gamma value to incorporate into new graphs
    running_list = [] #Will contain the population data

    for zombification in gammas:

        params = {**Part2_Params, "gamma": zombification} #Copies the dictionary but with gamma replaced
        times, populations = RK4(derivative_zombie, 0.0, TF_P2, STEP, params, state0_zombie)
        running_list.append(populations)

    runs = np.array(running_list) #contains the populations, now as an array. The times we use stays constant across gammas
    plot_gamma_variation('Gamma_Variation', times, runs, gammas) 

def run_outbreak_declaration():
    ''' Plotting the dz/dt over time. Printing the maximum dz/dt over time, 
        as well as printing the associated time and population levels.
    '''

    times, populations = RK4(derivative_zombie, 0.0, TF_P2, STEP, Part2_Params, state0_zombie)
    rates = derivative_zombie(times, Part2_Params, populations) #This returns an array will all 3 rates of change, with zombie rates are found at rates[1]
    plot_dz_by_dt('Zombie_Rate_over_time', rates[1], times) 

    max_dz_by_dt = max(rates[1]) #The largest value in the array

    outbreak_index = np.argmax(rates[1]) #This index at which we are declaring the outbreak

    outbreak_time = times[outbreak_index] #Calculates the associated time for when the zombie rate of change is maximum

    outbreak_humans     = int(np.round(populations[0][outbreak_index])) #The whole rounded number of humans at the time of the maximum rate of change of zombies 
    outbreak_zombies    = int(np.round(populations[1][outbreak_index])) #The whole rounded number of zombies at the time of the maximum rate of change of zombies
    outbreak_dead       = int(np.round(populations[2][outbreak_index])) #The whole rounded number of dead at the time of the maximum rate of change of zombies

    print(f'The maximum value for the rate of change of zombies with respect to time is: {max_dz_by_dt}, found at time {outbreak_time:.2f}.')
    print(f'Zombie outbreak declared at {outbreak_time:.2f}days, with {outbreak_humans} humans, {outbreak_zombies} zombies, and {outbreak_dead} dead.')

def run_outbreak_gammas():
    ''' Plotting outbreak time as a function of zombification rate, complete with a curve fit,
        and a residual. Prints the 2 fits as well as the maximum residuals for the two fits
    '''
    gammas = np.linspace(2e-3, 18e-3, 100) #Gammas equally spaced around a suitable range
    outbreak_times = [] #Will contain all the outbreak times

    for zombification in gammas:

        params = {**Part2_Params, "gamma": zombification} #Copies the dictionary but with gamma replaced
        times, populations = RK4(derivative_zombie, 0.0, TF_P2, STEP, params, state0_zombie)
        rates = derivative_zombie(times, params, populations)

        outbreak_times.append(times[np.argmax(rates[1])]) #Array of times for when rate of change of Z is at its max

    outbreak_times = np.array(outbreak_times) #Transforms list into array
    h = 0.01 #In days, the RK4 step size used in the sweep

    #First fit, T = a / gamma
    popt_a, pcov_a = curve_fit(outbreaks_model, gammas, outbreak_times, p0=[5e-3])
    perr_a = np.sqrt(np.diag(pcov_a)) #From the covariance matrix

    #Second fit, T = a / gamma**p
    popt_p, pcov_p = curve_fit(outbreaks_model_p, gammas, outbreak_times, p0=[5e-3, 1.0])
    perr_p = np.sqrt(np.diag(pcov_p))

    #Printing both of the fit results
    print(f'Fit 1: a = {popt_a[0]:.4e} +/- {perr_a[0]:.1e}')
    print(f'Fit 2: a = {popt_p[0]:.4e} +/- {perr_p[0]:.1e}, p = {popt_p[1]:.4f} +/- {perr_p[1]:.1e}')

    #Printed the greatest difference between the data and the fit. Grid error should be h/2
    print(f'Max residual, fit 1: {np.max(np.abs(outbreak_times - outbreaks_model(gammas, *popt_a))):.3f} days')
    print(f'Max residual, fit 2: {np.max(np.abs(outbreak_times - outbreaks_model_p(gammas, *popt_p))):.3f} days')

    plot_outbreaks('Outbreak_times_gamma_variation', gammas, outbreak_times, outbreaks_model_p, popt_p, h)

def run_parameter_adaptations():
    ''' Changes the different variables over time to model different situations and display these.'''

    delta0 = Part2_Params["delta"] #Initial zombie destruction rate
    delta_max = 200 * delta0 #Ceiling of zombie destruction rate

    params = {**Part2_Params, "delta_max": delta_max} #Adds ceiling to copied dictionary. Switches on the changing delta


    times, population = RK4(derivative_zombie, 0, 10.0, STEP, params, state0_zombie) #Hardcoded 10 days is suitable
    plot_zombie('delta_versatility', times, population)

def run_human_survival():
    ''' Changing the tau values over a range and plotting the Human survival rate as a result of that.
        The delta maximum is then overlaid onto this graph using a colour scale.
    '''

    delta0 = Part2_Params["delta"] #Initial zombie destruction rate
    delta_max = np.linspace(50, 500, 50) * delta0 #Ceiling of zombie destruction rate

    taus = np.arange(0.1, 2, 0.2) #Array of tau values which will be fed in to plot
    H_final = np.zeros((len(taus), len(delta_max))) #2D zero array storing H_final for every combination of parameters

    for i, tau in enumerate(taus):
        #loops through every tau
        for j, d_max in enumerate(delta_max):
            #For each tau goes through every delta_max
            params = {**Part2_Params, "tau": tau, "delta_max": d_max} #Updates the parameters to contain the current tau and delta_max
            times, populations = RK4(derivative_zombie, 0.0, 10.0, STEP, params, state0_zombie) #Hardcoded 10 days is more appropriate
            H_final[i, j] = populations[0, -1] #Extracts the final human population

    plot_taus('Human_Survival_Tau_Variation', taus, H_final, delta_max)


run_Euler_animals()
run_RK4_animals()
run_error_analysis()
run_RK4_zombies()
run_gamma_variation()
run_outbreak_declaration()
run_outbreak_gammas()
run_parameter_adaptations()
run_human_survival()

print('Program Finished')