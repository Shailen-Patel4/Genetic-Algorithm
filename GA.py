import math
import numpy as np
import matplotlib.pyplot as plt
import copy #you cannot just say a = z because this just means that a is a new reference to z, instead we acually have to copy the contents over from z to a and then we can change a without changing z
import random
import matplotlib.animation as animation

# Genetic Algorithm used to find the global minimum of the 5th order schwefel function

def binary2decimal(binary, Bits):
    decimal = 0
    for bit in range(0, Bits):
        if binary[bit] == 1:
            decimal += 2**(Bits - bit - 1)
        else:
            decimal += 0
    decimal1 = decimal- 524288
    decimal2 = decimal1/1000
    return decimal2


def random_population(pop_size, Bits, Order):
    random_pop = []
    for i in range(0, pop_size):
        random_chromosome = []
        for j in range(0,Order):
            random_chromosome.append(np.random.randint(0,2, size = Bits).tolist())
        random_pop.append([random_chromosome, 0, 0, 'parent/child'])
    return random_pop

def fitness(pop, Bits, Order): #calculates and updates the fitness of each member in the population
    for i in range(0,len(pop)):
        value = 0
        for j in range(0,Order):
            x  = binary2decimal(pop[i][0][j], Bits)
            value += x * math.sin(math.sqrt(math.fabs(x))) #maximising the negative of the schwefel function
        pop[i][1] = value

def combine_populations(Parents, Children): #combining everyone into one population and setting them up ready for tournament selection
    combined = copy.deepcopy(Parents) #taking a copy, if combined = Parents then this just creates a new ref to Parents and hence if combined is changed then so is Parents
    for i in children:
        combined.append(i)
    return combined

def parent_selection_probabilities(pop, S):
    N = len(pop)
    for i in range(0,N):
        R = i+1
        pop[i][2] = np.round((S*(N+1-2*R) + 2*(R-1))/(N*(N-1)),3)

def rank_population(pop): #sorting the list population with the second element (x[1]) of each x in population, using lambda here like a function with no name
    pop.sort(key = lambda x: x[1], reverse = True)

def roulette(pop, select):
    pop_copy = copy.deepcopy(pop)
    selected = []
    for j in range(0,select):
        p = np.random.rand()
        p_cum = 0
        for i in range(0,len(pop_copy)+1):
            if p > p_cum:
                if i == len(pop_copy): #protection against numerical errors when probabilities do not add to 1
                    selected.append(pop_copy[i-1])
                    break
                else:
                    p_cum += pop_copy[i][2]
            else:
                selected.append(pop_copy[i-1])
                constant = 1 - pop_copy[i-1][2]
                pop_copy.pop(pop_copy.index(pop_copy[i-1])) #removing chosen members
                for i in pop_copy:
                    i[2] = i[2]/constant # normalising the probabilities
                break
    for i in selected:
        i[3] = "parent"
    return selected





### USING TOURNAMENT SELECTION ###
# The function receives a set number of parents which is always set = population_size
# the size of children is variable and hence we need to dynamicaly change the size of the teams.
# To do be able to accommodate this, I have chosen to produce n whole teams and the remaining members are
# combined with the last team meaning that the last team will always be team_size or greater.
# Then, calculate the number of members to be chosen from each team to make up pop_size.
def tournament_selection(Parents, Children, Bits, Order, pop_size, team_size, S):
    combined_population = combine_populations(Parents, Children)
    fitness(combined_population, Bits, Order)
    selection_indexes = []
    for i in range(0,len(combined_population)):
        selection_indexes.append(i)
    teams = len(combined_population)//team_size  # number of whole teams

    if len(combined_population) % team_size == 0:
        last_team_size = team_size
    else:
        last_team_size = len(combined_population) - ((teams - 1) * team_size)

    team_indexes = [] #list of lists of the indexes of the members in each team
    team_members = []
    selected_members = []
    next_generation = []

    # setting up random teams
    for i in range(0, teams - 1):
        team_indexes.append(random.sample(selection_indexes, team_size))
        team_members.append([])
        for j in team_indexes[i]: # removing the members that have been chosen for a team to stop chosing the same member
            selection_indexes.pop(selection_indexes.index(j))
            team_members[i].append(combined_population[j])
        rank_population(team_members[i])
        parent_selection_probabilities(team_members[i], S)

        if i == teams - 2: #for appending the last team
            team_indexes.append(random.sample(selection_indexes, last_team_size))
            team_members.append([])
            for j in team_indexes[i+1]:
                selection_indexes.pop(selection_indexes.index(j))
                team_members[i+1].append(combined_population[j])

            rank_population(team_members[i+1])
            parent_selection_probabilities(team_members[i+1], S)

    team_members.reverse()# reversing the list so the potential largest team is always part of the larger groups

    select = pop_size // teams #lower bound on number of members to select per team to ensure a total of pop_size members have been picked
    larger_groups = pop_size - select * teams
    for i in range(0, teams):
        if i < larger_groups:
            selected_members.append(roulette(team_members[i], select + 1))
        else:
            selected_members.append(roulette(team_members[i], select))
    # combining all selected members from all teams into next_generation
    for i in selected_members:
        for j in i:
            next_generation.append(j)

    return next_generation


def breeder(parent1, parent2, Order): #This function contains a parameter that we can tune to optimise the algorithms performance
    genes = []
    crossover1 = int(len(parent1[0][0])*0.25) # arbitrarily chosen these crossover points
    crossover2 = int(len(parent1[0][0])*0.75)
    for i in range(0,Order):
        for j in range(crossover1, crossover2): # crossing over all the genes in both parents
            x = parent1[0][i][j]
            parent1[0][i][j] = parent2[0][i][j]
            parent2[0][i][j] = x
    return parent1, parent2

def breed(parents, Pc, Bits, Order):
    parents_copy = copy.deepcopy(parents)
    random.shuffle(parents_copy) # randomly selecting breeding pairs
    N = len(parents_copy)
    children = []
    for i in range(0, int(N/2)):
        p = np.random.rand()
        if p <= Pc:
            (child1, child2) = breeder(parents_copy[i], parents_copy[i+int(N/2)], Order)

            child1[3] = "child"
            child2[3] = "child"

            children.append(child1)
            children.append(child2)
    fitness(children, Bits, Order) # calculating the fitness of the new children
    return children

def mutate(children, Bits, Order, Pm):
    for i in range(0,len(children)):
        for j in range(0, Order):
            for k in range(0, Bits):
                p = np.random.rand()
                if p <= Pm:
                    if children[i][0][j][k] == 1: # flipping the bits
                        children[i][0][j][k] = 0
                    else:
                        children[i][0][j][k] = 1
    fitness(children, Bits, Order)

def check(Parents, Children, Bits, Order, Previous_solutions, tolerance):
    combined = combine_populations(Parents, Children)
    fitness(combined, Bits, Order)
    rank_population(combined)

    Previous_solutions.append(combined[0]) # storing all the previous best members of the population

    if len(Previous_solutions) > 50: # this requires that at least 50 generations have passed before it can be terminated
        Previous_solutions.pop(0)
        if np.absolute(Previous_solutions[0][1] - Previous_solutions[-1][1]) <= tolerance: # comparing the fitness of the current best member to the best member 50 generations ago
            return True
        else:
            return False
    else:
        return False




###   Parameters   ###
(bits, order) = (20, 2) # 20 = max number of bits to represent +-500,000 but in function we *0.001 which allows the answer to be accurate to 3dp (balance of precision over runtime) and order of schwefel function
(population_size, P_crossover, P_mutation) = (500, 1, 0.01)
selection_pressure = 1.3 # 1<=S<=2 where S=1 gives uniform (the choice of s = 1.3 was experimentally determined after testing s = {1,1.5,2})
previous_solutions = [] # variable to store the last several generations best solutions
tolerance = 0.01 # tolerance to stop the algorithm
timeout = 5000 # max number of iterations the algorithm will run if the error is not minimised to the tolerance
tournament_team_size = 5

###   Generating the initial population   ###
parents = random_population(population_size, bits, order)
children = random_population(population_size, bits, order)

plot_data = [] # variable to store the data required for plotting

for iter in range(0, timeout):
    selected_parents = tournament_selection(parents, children, bits, order, population_size, tournament_team_size, selection_pressure) # make sure that the team size is valid with the number of parents and children
    new_children = breed(selected_parents, P_crossover, bits, order)
    mutate(new_children, bits, order, P_mutation)

    children = copy.deepcopy(new_children) # updating children
    parents = copy.deepcopy(selected_parents) # updating parents

    if order == 2: #only for plotting the second order schwefel function
        plot_data.append([]) #storing the data to plot
        plot_data[iter] = []
        for i in parents:
            plot_data[iter].append(i[0])

    status = check(parents, children, bits, order, previous_solutions, tolerance)

    if status == True:
        break


print(iter) # printing the final generation number

# For printing the top 10 members of the final generation #
combined = combine_populations(parents, children)
fitness(combined, bits, order)
rank_population(combined)

counter = 1
for i in combined[0:10]:
    print("position: %d" %counter)
    counter += 1
    print(i[1])
    for j in range(0,order):
        print("solution: %d" %(j+1))
        print(binary2decimal(i[0][j],bits))



### plotting the evolution graph ###
# plotting the evolution as an animation to best visualise how the distribution of the population shifts as it evolves

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

    scatter_plot_template, = plt.plot([], [], 'r*')

    iteration_template = 'Iteration = %d' #this is a string template where %f means a floating point number and %.1f means that floating point number is rounded to a precision of 1
    iteration_text = plt.text(-450, -450, '', fontsize = 15, bbox = dict(facecolor='blue', alpha=0.5))#alpha is the transparency of the box i.e. meaning that alpha = 1 is opaque
    def animate(i):
        x1 = []
        x2 = []
        for j in plot_data[i]:
            x1.append(binary2decimal(j[0], bits))
            x2.append(binary2decimal(j[1], bits))
        scatter_plot_template.set_data(x1, x2)
        iteration_text.set_text(iteration_template % (i+1))
        return scatter_plot_template, iteration_text,

    def init():
        plt.xlim(-500,500)
        plt.ylim(-500,500)
        scatter_plot_template.set_data([],[])
        iteration_text.set_text('')
        return scatter_plot_template, iteration_text,


    ani = animation.FuncAnimation(fig, animate, init_func = init, frames = np.arange(0, iter+1), interval = 100, repeat = True, blit = True, repeat_delay = 1000)

    plt.show()
    plt.close(fig)
