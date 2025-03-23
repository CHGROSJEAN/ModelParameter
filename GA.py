# ==============================================================================
# Script Title: Model Parameter Estimation - Genetic Algorithm
# Author: Charlotte Grosjean & Joanne Quet
# Date: March 2025
# Description:
#   This script implements a genetic algorithm to optimize a population of individuals 
#   with porosity values, evolving over multiple generations to match observed data based 
#   on a likelihood fitness function. Each new generation is created based on a selection 
#   of individuals for breeding, random pairing of parents DNA, and mutation to create children,
#   along with immigration of new randomly generated individuals.
#   The objective of the algorithm is to improve the population's fitness across generations.
#   
# ==============================================================================

import numpy as np
import scipy.io
import math as m
import random as rd
import matplotlib.pyplot as plt

#%% INITIALISATION

## Loading necessary data
content = scipy.io.loadmat('data.mat') # This just loads a dictionary
dataobs = content['dataobs10'] # Select dataset
A = content['A']

## Parameters
nx = 40 # horizontal coordinate 
ny = 60 # vertical coordinate 
N = 36 # number of datapoint
minpor = 0.2 # minum porosity
maxpor = 0.4 # maximum porosity
sigma = 1 # uncertainty sigma used in the likelyhood calculation

ks = 5 # permitivity of grains 
kw = 81 # permitivity of water
c = 0.3 # celerity of light

nb_ind = 100 # number of individuals in each generation

nb_breed = 20 # number of the fittest individuals chosen for breeding
# here we select the 20% first fittest individuals

nb_gen = 50 # number of generations

mutation_rate = 0.1 # mutation rate

nb_migration = 10 # number of new individuals from immigration

#%% FIRST GENERATION CREATION

# Matrix for porosity values of Generation 1
Gen1 = np.zeros((nb_ind,4)) 

# Matrix for likelihoods of all Generations
L_tot = np.zeros((nb_ind,nb_gen))
   
# Drawing random porosity values for each individual            
for i in range (nb_ind):
    Gen1[i, :] = np.random.uniform(minpor, maxpor, 4) 
    


# Compute the fitnes of each individual of a generation using the likelihood function
def likelihood_fitness(Gen, nb_ind): # creation of a function to be used later on
    L_gen = np.zeros(nb_ind) # Initialise one vector for the likelihood of the generation 
    
    for i in range(nb_ind):
        # Convert the porosity values in slowness
        Kroots = np.zeros(4)
        for j in range(4):
            Kroots[j] = (1-Gen[i,j])* (m.sqrt(ks)) + Gen[i,j]*(m.sqrt(kw))
            
        S = Kroots/c
    
        # Apply the value of the posority for all the layer pixels.
        sgrid = np.zeros((nx,ny))
        sgrid[0:nx, 0:20]=S[0]
        sgrid[0:nx,20:30]=S[1]
        sgrid[0:nx,30:50]=S[2]
        sgrid[0:nx,50:60]=S[3]
        sarray = np.reshape((np.transpose(sgrid)),(2400,1))
        
        # Compute our synthetic data
        d_syn = A@sarray
    
        # Compute the likelihood of the created individual
        L_gen[i] = (1/((m.sqrt(2*m.pi)**N)*sigma**N))*m.exp(-0.5*(((np.transpose((d_syn-dataobs)/sigma))@((d_syn-dataobs)/sigma))[0,0]))
        
        
        # Check that the Likelihood is not null
        if L_gen[i] == 0:
            L_gen[i] = 10**(-300)
            
    return L_gen
    
# Fitness for the individuals of generation 1 stored into L_tot
L_tot[:,0] = likelihood_fitness(Gen1, nb_ind)

print(Gen1)

# je definis avant les fonctions à utiliser puis je les utilise dans la dernière "breeding"
#%% SELECTION FOR BREEDING

def breeding_selection(L, Gen, nb_breed):
    sorted_L = np.argsort(L)[::-1] # sort the Likelihoods from the largest to the smallest and get the indices
    select_ind = sorted_L[:nb_breed] # select the indices of the nb_breed fittest individuals
    fittest_ind = Gen[select_ind] # select only the fittest individual in the generation with their indices
    first_ind = Gen[sorted_L[0]] # select the individual with the maximum likelihood
    
    # Print sorted likelihoods and the corresponding individuals for verification
    print("Sorted likelihoods for generation:")
    print(L[sorted_L])  # Print the sorted likelihood values
    print("Corresponding individuals with sorted likelihoods:")
    print(fittest_ind)  # Print the best individuals corresponding to the sorted likelihoods

    return fittest_ind, first_ind

#%% ENCODING DNA
def encodingDNA(Gen):
    Gen_string = np.zeros(len(Gen), dtype=object)  # Initializing a matrix to store DNA as strings
    for i in range(len(Gen)):
        # Ensure that Gen[i] is a 1D array with scalar values
        Gen_string[i] = "{:.3f}{:.3f}{:.3f}{:.3f}".format(
            float(Gen[i, 0]) if isinstance(Gen[i, 0], (int, float)) else float(Gen[i, 0][0]),
            float(Gen[i, 1]) if isinstance(Gen[i, 1], (int, float)) else float(Gen[i, 1][0]),
            float(Gen[i, 2]) if isinstance(Gen[i, 2], (int, float)) else float(Gen[i, 2][0]),
            float(Gen[i, 3]) if isinstance(Gen[i, 3], (int, float)) else float(Gen[i, 3][0])
        )
    return Gen_string

#%% DNA MIXING

def DNAmix(parents):
    cut_point = rd.randint(1, len(parents[0]) - 1) # Choose a random cut point
    # Create children DNA by combining first part of first parent DNA to second part of second parent DNA
    child = parents[0][:cut_point] + parents[1][cut_point:]
    return child

#%% MUTATION

def mutation(child, mutation_rate):
    """
    Mutates a vector of DNA strings with a given mutation rate.
    
    Parameters:
    - child: List or array of DNA strings (e.g., ['0.3820.2400.2390.329', ...]).
    - mutation_rate: Probability that a child's DNA will be mutated.
    
    Returns:
    - A mutated vector of DNA strings.
    """
    child_mutated = []
    for DNA in child:
        # Decide whether to mutate this child's DNA based on mutation rate
        if rd.random() < mutation_rate:
            # Convert DNA string to a list of characters for modification
            mutation = list(DNA)
            
            # Choose a random index to mutate, excluding decimal points
            mutable_indices = [i for i in range(len(DNA)) if i % 5 != 1]  # Exclude decimal points at indices 1, 6, 11, etc.
            if mutable_indices:
                j = rd.choice(mutable_indices)  # Pick a random index to mutate
                mutation[j] = str(rd.randint(0, 9))  # Replace with a random digit
            
            # Convert back to string after mutation
            mutated_DNA = "".join(mutation)
        else:
            # No mutation, keep the original DNA
            mutated_DNA = DNA
        
        # Add the mutated or original DNA to the result list
        child_mutated.append(mutated_DNA)
    
    return np.array(child_mutated, dtype=object)


#%% DECODING DNA

def decodingDNA(NewGen):
    """
    Decodes a list of DNA strings into a NumPy array of porosity values.
    
    Parameters:
    - NewGen: List or array of DNA strings (e.g., ['0.2660.3110.2940.318', ...]).
    
    Returns:
    - A NumPy array of decoded porosity values.
    """
    Gen_decoded = []  # List to store decoded matrix rows
    for DNA in NewGen:
        try:
            # Ensure the DNA string has the correct length (20 characters for 4 floats)
            if len(DNA) != 20:
                raise ValueError(f"Invalid DNA length: {DNA}")

            # Split the DNA string into 4 substrings of length 5 (each representing one porosity value)
            porosities = [float(DNA[i:i+5]) for i in range(0, len(DNA), 5)]  # Convert each substring to a float
            
            # Append the porosity values to the list
            Gen_decoded.append(porosities)
        except ValueError as e:
            print(f"Error decoding DNA: {DNA} - {e}")
            continue  # Skip invalid DNA strings
    
    if not Gen_decoded:  # Ensure we don't return an empty array
        raise ValueError("All decoded DNAs are invalid.")
    
    return np.array(Gen_decoded)  # Return as a NumPy array



#%% CHECK PRIOR

def check_prior(Gen_decoded, minpor, maxpor):
    """
    Check if the decoded porosity values are within the prior range.
    Returns the updated Gen_decoded and a list indicating whether each individual is within the prior range.
    """
    within_prior = []  # List to track whether the individuals' porosity values are within the prior range
    for individual in Gen_decoded:
        # Check if all values of the individual are within the prior range
        within_prior_individual = [minpor <= value <= maxpor for value in individual]
        within_prior.append(within_prior_individual)
        
        # If any value is outside the prior, do not modify the individual (keep it as is)
        # This assumes you don't modify individuals in this function, only track whether they're valid
    return Gen_decoded, within_prior

#%% BREEDING

def breeding(nb_ind, Gen, L_tot, nb_breed, mutation_rate, minpor, maxpor, generation):
    """
    Performs breeding to generate a new generation of individuals.
    
    Parameters:
    - nb_ind: Number of individuals in each generation.
    - Gen: Current generation matrix.
    - L: Likelihood matrix for all generations.
    - nb_breed: Number of fittest individuals selected for breeding.
    - mutation_rate: Mutation rate applied to children.
    - minpor, maxpor: Minimum and maximum porosity values for prior checks.
    - generation: The current generation index (to select the best individual from the previous generation).
    
    Returns:
    - NewGen: Matrix of valid children generated through breeding and mutation.
    """
    NewGen = np.zeros((nb_ind, 4), dtype=object)  # Initialize new generation matrix

    # Clone the best individual from the previous generation
    best_ind = breeding_selection(L_tot[:, generation - 1], Gen, nb_breed)[1]
    NewGen[0] = np.round(best_ind, 3)  # Add the best individual to the new generation

    # Retrieve only the fittest individuals and encode their DNA
    fittest_ind = breeding_selection(L_tot[:, generation - 1], Gen, nb_breed)[0]
    fittest_list = encodingDNA(fittest_ind)

    # Shuffle the fittest individuals randomly
    rd.shuffle(fittest_list)
    
    # Pair shuffled individuals for DNA mixing
    pairs = [(fittest_list[i], fittest_list[i + 1]) for i in range(0, len(fittest_list) - 1, 2)]

    valid_children = []  # List to store valid children

    # Generate valid children through DNA mixing and mutation
    while len(valid_children) < nb_ind - 1 - nb_migration:  # Leave space for the best individual
        for parent1, parent2 in pairs:
            child = DNAmix((parent1, parent2))  # Create a child from parents' DNA
            mutated_child = mutation([child], mutation_rate)  # Apply mutation
            
            try:
                decoded_child = decodingDNA(mutated_child)  # Decode mutated DNA
                decoded_child_within_prior, within_prior_flags = check_prior(decoded_child, minpor, maxpor)

                if all(within_prior_flags[0]):  # Check if child is within prior range
                    valid_children.append(decoded_child[0])  # Add valid child to list
                    if len(valid_children) >= nb_ind - 1- nb_migration:
                        break

            except ValueError as e:
                print(f"Skipping invalid child: {e}")
                continue

    # Add valid children to NewGen
    NewGen[1:len(valid_children) + 1] = valid_children[:nb_ind - 1]

    return NewGen

#%% MIGRATION

def migration(nb_migration, minpor, maxpor):
    # Generate new individuals to migrate into the population
    immigration = np.zeros((nb_migration, 4), dtype=object)
    for i in range(nb_migration):
        immigration[i, :] = np.random.uniform(minpor, maxpor, 4) # Generate random porosity values for each individual  
    return immigration

#%% POPULATION EVOLUTION

# Initialize a 3D matrix to hold all generations
Pop = np.zeros((nb_gen, nb_ind, 4), dtype=float)
Pop[0] = Gen1  # Store the first generation in the 3D matrix

# Initialize a matrix to hold the new generation
NewGen = np.zeros((nb_ind, 4), dtype=object) 

# Initialize the current generation with the first generation
Gen = Gen1

# Main loop for population evolution
for generation in range(1, nb_gen):  # Start from generation 1 since Gen1 is already initialized
    print(f"Generation {generation}/{nb_gen}")

    # Step 1: Perform Breeding
    NewGen = breeding(nb_ind - nb_migration, Gen, L_tot, nb_breed, mutation_rate, minpor, maxpor, generation)  # Pass current generation index

    # Step 2: Apply Migration
    immigration = migration(nb_migration, minpor, maxpor)  # Generate new immigrants
    
    # Combine bred individuals with immigrants
    CombinedGen = np.vstack((NewGen[:], immigration))

    # Step 3: Calculate likelihoods for CombinedGen
    likelihoods_new_gen = likelihood_fitness(CombinedGen.astype(float), nb_ind)
    
    # Update population and likelihoods for next generation
    Pop[generation] = CombinedGen
    L_tot[:, generation] = likelihoods_new_gen

# Final population stored in Pop and likelihoods in L_tot

#%% PLOTTING

# Example: Plotting best fitness over generations
best_fitness = np.max(L_tot, axis=0)  # Best fitness in each generation
plt.plot(range(nb_gen), best_fitness)
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Convergence Plot")
plt.show()




# Example: Plotting population diversity over generations
diversity = np.std(L_tot, axis=0)  # Standard deviation of fitness in each generation
plt.plot(range(nb_gen), diversity)
plt.xlabel("Generation")
plt.ylabel("Fitness Diversity (Standard Deviation)")
plt.title("Population Diversity Over Generations")
plt.show()



# Initialize the figure with 4 subplots (one for each parameter)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid of subplots
axs = axs.flatten()  # Flatten to iterate over all axes

# Iterate over each parameter (0 to 3 for porosity layers)
for param_idx in range(4):
    mean_values = np.mean(Pop[:, :, param_idx], axis=1)  # Mean porosity per generation
    std_values = np.std(Pop[:, :, param_idx], axis=1)    # Standard deviation per generation
    best_values = np.zeros(nb_gen)   # Best individual per generation, initialize with zeros

    # Find the best individual (highest likelihood) for each generation
    for gen in range(nb_gen):
        best_idx = np.argmax(L_tot[gen])  # Get the index of the highest likelihood in generation `gen`
        best_values[gen] = Pop[gen, best_idx, param_idx]  # Get the corresponding porosity value for that index


    generations = np.arange(nb_gen)  # Generation numbers

    # Plot mean porosity
    axs[param_idx].plot(generations, mean_values, label="Mean Porosity", color="blue")
    
    # Plot best individual porosity
    axs[param_idx].plot(generations, best_values, label="Best Individual", color="green", linestyle="--")
    
    # Plot mean ± std
    axs[param_idx].fill_between(
        generations,
        mean_values - std_values,
        mean_values + std_values,
        color="blue",
        alpha=0.2,
        label="Mean ± Std"
    )

    # Customize subplot
    axs[param_idx].set_title(f"Parameter {param_idx + 1}")
    axs[param_idx].set_xlabel("Generation")
    axs[param_idx].set_ylabel("Porosity")
    axs[param_idx].legend()
    axs[param_idx].grid()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

pass



# Compute the mean likelihood for each generation (across all individuals)
mean_likelihood = np.mean(L_tot, axis=0)  # Mean likelihood per generation

# Create a plot
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, nb_gen + 1), mean_likelihood, label="Mean Likelihood", color="blue", marker="o")

# Customize the plot
plt.title("Evolution of Mean Likelihood Over Generations")
plt.xlabel("Generation")
plt.ylabel("Mean Likelihood")
plt.grid(True)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()




# Initialize an array to store the best likelihood for each generation
best_likelihood = np.zeros(nb_gen)

# Loop through each generation and find the best individual (highest likelihood)
for generation in range(nb_gen):
    best_likelihood[generation] = np.max(L_tot[:, generation])  # Store the maximum likelihood of each generation

# Plot the best likelihood for each generation
plt.plot(range(nb_gen), best_likelihood, label="Best Individual Likelihood", color="green", linestyle="--")
plt.xlabel("Generation")
plt.ylabel("Likelihood")
plt.title("Best Individual Likelihood Across Generations")
plt.grid(True)
plt.legend()
plt.show()





