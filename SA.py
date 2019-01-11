import math
import numpy as np
import matplotlib.pyplot as plt
import copy #cannot say a = z because this just means that a is a new reference to z, instead we acually have to copy the contents over from z to a and then we can change a without changing z
import random
import matplotlib.animation as animation

# Simulated Annealing algorithm to minimise the schwefel function
# based on matlab function "General simulated annealing algorithm" by Joachim Vandekerckhove

def trial(Order): # used to generate u in a new solution in x_i+1 = x_i + Du
    return np.random.uniform(-1, 1, Order)


def f(x, Order):
    f = 0.0
    for i in range(0, Order):
        f += -x[i] * math.sin(math.sqrt(math.fabs(x[i])))
    return f

def d_f(x_init, x_try, Order): # compares the new solution to the initial one and returns the change in f
    return f(x_try, Order) - f(x_init, Order)

def update_D(D_prev, Alpha, Omega, U, Order):
    R = np.eye(Order)
    for i in range(0, Order):
        R[i,i] = math.fabs(D_prev[i,i]*U[i])
    D_next = (1 - Alpha)*D_prev + Alpha*Omega*R
    return D_next

def check_T(T, alpha_T, markov_length, same_solution_counts, min_acceptances, acceptance_counter):
    if same_solution_counter >= markov_length:
        T_new = alpha_T * T
        return True, T_new
    elif acceptance_counter >= min_acceptances:
        T_new = alpha_T * T
        return True, T_new

    return False, 0.0


def calculate_step_size(D, U, Order):
    R = np.eye(Order)
    for i in range(0, Order):
        R[i,i] = math.fabs(D[i,i]*U[i])
    d_square = 0
    for i in range(0, Order):
        d_square += np.square(R[i,i])
    return math.sqrt(d_square)


def accept(Df, temp, d):
    p = math.exp(-((Df)/(d*temp))) # calculating the probability of accepting the solution
    sample = np.random.rand() # sampling uniformly in [0.0, 0.1)
    if sample <= p:
        return True # accepted sample
    else:
        return False # rejected sample
    return False




def initial_temp(x_init, D_init, Order):
    df_counter = 0 # counts the number of times df is positive
    N = 10000 # total number of trials to take the average over
    for i in range(0,N):
        x_try = np.matmul(D_init,trial(Order))
        df = d_f(x_init, x_try, Order)
        if df > 0:
            df_counter += 1

    df_avg = df_counter / N
    t0 = - df_avg / math.log(0.8) # Kirkpatrick [1984] using p = 0.8
    return t0*10 #scaling by 10 to allow for a bigger search range for the first few iterations


# Parameters
order = 5
alpha = 0.1 # damping constant
omega = 1.7 # weighting on rate at which R transfers to D
temperature_decrement_constant = 0.9 # Kirkpatrick [1982]
max_attempts = 300
max_accepts = 30
stop_temp = 0.1
max_consec_rejections = 100

# counters and tracking variables
accepts = 0 # success
attempts = 0 # itry
consecutive_rejections = 0
finished = False
total = 0
first_time = True

D = np.eye(order) * 150 # setting the initial diagonal matrix to contain the maximum range of -500 <= x_i <= 500
x_initial = np.matmul(D, trial(order)) # initial random starting solution (but can initialise with a specific solution)
T = initial_temp(x_initial, D, order) # initial temperature calculated using Kirkpatrick [1984]

initial_energy = f(x_initial, order)
old_energy = copy.copy(initial_energy)

plot_data = []

accepted_solution = copy.copy(x_initial)

while finished == False:
    attempts += 1
    current = copy.deepcopy(accepted_solution)
    plot_data.append(current)

    # adjusting the temperature if required
    if attempts >= max_attempts or accepts >= max_accepts: #checking if the criteria for decrementing T is met
        if T <= stop_temp or consecutive_rejections >= max_consec_rejections:
            finished = True
            total = total + attempts
            attempts = 1
            accepts = 1
            break

        else:
            T_next = temperature_decrement_constant * T
            T = copy.copy(T_next)
            total += attempts
            attempts = 1
            accepts = 1

    # generating a new solution
    u = trial(order)
    new_solution = current + np.matmul(D, u)

    for i in range(0, new_solution.size):
        if new_solution[i] > 500.0:
            new_solution[i] = 500.0
        elif new_solution[i] < -500.0:
            new_solution[i] = -500.0


    new_energy = f(new_solution, order)
    df = new_energy - old_energy

    #assessing the new solution
    if df < 0: # i.e. checking if the solution decreases the function
        first_time = False
        accepted_solution = copy.deepcopy(new_solution) # accepting the new solution
        D = update_D(D, alpha, omega, u, order)
        old_energy = copy.copy(new_energy)
        accepts += 1
    else: # new solution increases the function
        if first_time == True: # for the first iteration the step size d cannot be calculated as there is no accepted solution yet
            first_time = False
            d = 1
        else:
            d = calculate_step_size(D, u, order)

        sample = np.random.rand()



        p = math.exp(-((df)/(d*T)))

        if sample <= p:
            accepted_solution = copy.deepcopy(new_solution) # accepting the new solution
            D = update_D(D, alpha, omega, u, order)
            old_energy = copy.copy(new_energy)
            accepts += 1
            consecutive_rejections = 0
        else:
            consecutive_rejections += 1




minimum_solution = accepted_solution
fval = old_energy


print("minimum solution found is ", minimum_solution)
print("minimum energy found is", fval)
print("total number of iterations is", total)
print("final temperature = ", T)


if order == 2:

    #data to plot the schwefel contours
    number_of_points = 200
    schwefel_x1_list = np.linspace(-500, 500, number_of_points)
    schwefel_x2_list = np.linspace(-500, 500, number_of_points)
    (schwefel_x1, schwefel_x2) = np.meshgrid(schwefel_x1_list, schwefel_x2_list)
    schwefel_val = -schwefel_x1 * np.sin(np.sqrt(np.absolute(schwefel_x1))) -schwefel_x2 * np.sin(np.sqrt(np.absolute(schwefel_x2)))

    w = 10
    h = 10
    d = 70
    fig = plt.figure(figsize = (w, h), dpi = d)

    plt.xlabel("x1")
    plt.ylabel("x2")
    title_template = "Second order Schwefel function plot of evolving data"
    plt.title(title_template)


    cp = plt.contourf(schwefel_x1, schwefel_x2, schwefel_val)
    plt.colorbar(cp)

    scatter_plot_template, = plt.plot([], [], 'ro-')

    iteration_template = 'Iteration = %d' #this is a string template where %f means a floating point number and %.1f means that floating point number is rounded to a precision of 1
    iteration_text = plt.text(-450, -450, '', fontsize = 15, bbox = dict(facecolor='blue', alpha=0.5))#alpha is the transparency of the box i.e. meaning that alpha = 1 is opaque
    def animate(i):
        x1 = []
        x2 = []
        for k in range(0,i):
            x1.append(plot_data[k][0])
            x2.append(plot_data[k][1])

        scatter_plot_template.set_data(x1, x2)
        iteration_text.set_text(iteration_template % (i+1))
        return scatter_plot_template, iteration_text,

    def init():
        plt.xlim(-500,500)
        plt.ylim(-500,500)
        scatter_plot_template.set_data([],[])
        iteration_text.set_text('')
        return scatter_plot_template, iteration_text,


    ani = animation.FuncAnimation(fig, animate, init_func = init, frames = np.arange(0, total+1), interval = 5, repeat = True, blit = True, repeat_delay = 1000)
    plt.show()
    plt.close(fig)
