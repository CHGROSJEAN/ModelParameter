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

nb_gen = 30 # number of generations

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


# je definis avant les fonctions à utiliser puis je les utilise dans la dernière "breeding"
#%% SELECTION FOR BREEDING

def breeding_selection(L, Gen, nb_breed):
    sorted_L = np.argsort(L)[::-1] # sort the Likelihoods from the largest to the smallest and get the indices
    select_ind = sorted_L[:nb_breed] # select the indices of the nb_breed fittest individuals
    fittest_ind = Gen[select_ind] # select only the fittest individual in the generation with their indices
    first_ind = Gen[sorted_L[0]] # select the individual with the maximum likelihood
    return fittest_ind, first_ind

#%% ENCODING DNA

def encodingDNA(Gen):
    Gen_string = np.zeros(len(Gen), dtype=object) # Initialising a matrix that can take strings
    for i in range(len(Gen)):
        Gen_string[i] = "{:.3f}{:.3f}{:.3f}{:.3f}".format(Gen[i, 0], Gen[i, 1], Gen[i, 2], Gen[i, 3])
        # we round each value to 3 decimals, concatenate the result and transform it to strings
    return Gen_string

#%% DNA MIXING

def DNAmix(parents):
    cut_point = rd.randint(1, len(parents[0]) - 1) # Choose a random cut point
    # Create children DNA by combining first part of first parent DNA to second part of second parent DNA
    child = parents[0][:cut_point] + parents[1][cut_point:]
    return child

#%% MUTATION

def mutation(child, mutation_rate): # Function to mutate the DNA of a child
    child_mutated = []
    forbidden_indices = {0, 1, 5, 6, 10, 11, 15, 16} # Define the indices that cannot be mutated
    for DNA in child:  # Iterate through all children
        mutation = list(DNA)  # Convert string to list for modification
        mutable_indices = [i for i in range(len(DNA)) if i not in forbidden_indices]  # Allowed mutation indices
        if mutable_indices and rd.random() < mutation_rate:  # Apply mutation with mutation_rate probability
            j = rd.choice(mutable_indices)  # Pick a random index to mutate
            mutation[j] = str(rd.randint(0, 9))  # Replace that character with a random digit

        child_mutated.append("".join(mutation))  # Convert back to string and store
    return child_mutated  # Return the list of children after mutation

#%% DECODING DNA
Gen_decoded = []  # List to store decoded matrix rows

def decodingDNA(NewGen):
    for DNA in NewGen:
        # Split the DNA string into 4 substrings of length 4 (each representing one porosity value)
        porosities = [float(DNA[i:i+4]) for i in range(0, len(DNA), 4)] # Convert each substring to a float
        Gen_decoded.append(porosities)  # Append the porosity values to the list
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

def breeding(nb_ind, Gen, L, nb_breed, Gen_decoded, mutation_rate, nb_migration, minpor, maxpor):

    for gen_index in range(1, nb_gen):  # Loop through each generation

        # Initiatlisation of a matrix for new Generation 
        NewGen = np.zeros((nb_ind,4), dtype=object)
        # Cloning of the best individual from Gen 1 in Gen 2 and encode his DNA
        best_ind = breeding_selection(L, Gen, nb_breed)[1]
        best_ind = np.expand_dims(best_ind, axis=0) # Transform the vector in 2D, so it has the shape 1 x 4
        NewGen[0, :] = encodingDNA(best_ind)
        # Retreive only the fittest individuals and encode their DNA
        fittest_ind = breeding_selection(L, Gen, nb_breed)[0]
        fittest_list = encodingDNA(fittest_ind)
        # Shuffle the individuals randomly
        rd.shuffle(fittest_list)
        pairs = [(fittest_list[i], fittest_list[i+1]) for i in range(0, len(fittest_list)-1, 2)] # Pair the shuffled individuals
        # For each pair create children by mixing the parents DNA
        
        index = 1  # Start index for NewGen after the best individual
        valid_children = 0  # Counter for valid children
        
        while valid_children < nb_ind - nb_migration:  # Keep generating until we have the desired number of individuals
            for parent1, parent2 in pairs:
                child = DNAmix((parent1, parent2)) # Create a child from the parents' DNA
                # Apply mutation to the children
                mutated_child = mutation(child, mutation_rate)
                # Decode the DNA of the children
                Gen_decoded = decodingDNA(mutated_child)
                # Check if the children are within the prior range
                Gen_decoded, within_prior = check_prior(Gen_decoded, minpor, maxpor)
                
                # If the child is within the prior range, add it to NewGen
                if all(within_prior[index]):  # Ensure all the values of the child are within the prior range
                    NewGen[index, :] = mutated_child
                    valid_children += 1 # Increment the valid children counter
                    index += 1  # Move to the next position in NewGen
                    
                    # calculate the likelihood of the child
                    L_tot[index, gen_index] = likelihood_fitness(Gen_decoded, nb_ind)  # Add the likelihood of the child to the matrix
                    
                if valid_children >= nb_ind - nb_migration:
                    break
        
        # Update the current population for the next generation
        Gen = NewGen  # The new generation becomes the current generation for the next loop
    
    return Gen

#%% MIGRATION

def migration(nb_migration, minpor, maxpor):
    # Generate new individuals to migrate into the population
    immigration = np.zeros((nb_migration, 4), dtype=object)
    for i in range(nb_migration):
        immigration[i, :] = np.rd.uniform(minpor, maxpor, 4) # Generate random porosity values for each individual  
    return immigration

#%% POPULATION EVOLUTION

# Initialize a 3D matrix to hold all generations
Pop = np.zeros((nb_gen, nb_ind, 4), dtype=object)
Pop[0] = Gen1  # Store the first generation in the 3D matrix

# Initialize a matrix to hold the new generation
NewGen = np.zeros((nb_ind, 4), dtype=object) 

# Initialize the current generation with the first generation
Gen = Gen1

# Loop through each generation
for generation in range(nb_gen):
    print(f"Generation {generation + 1}/{nb_gen}")

    # Step 1: Perform Breeding
    NewGen = breeding(nb_ind, Gen, L_tot, nb_breed, Gen_decoded, mutation_rate, nb_migration, minpor, maxpor)  # Generate new individuals via breeding

    # Step 2: Apply Migration
    immigration = migration(NewGen, nb_migration)  # Generate new immigrants
    NewGen = np.vstack((NewGen, immigration))  # Add the immigrants to the new generation

    # Step 3: Store the new generation (2D matrix) into the 3D matrix
    Pop[generation] = NewGen  # Store the 2D matrix for the current generation




# We call the breeding function to create Gen 2 from Gen 1
#Gen2 = breeding(nb_ind, Gen1, L_tot[:,0], nb_breed)

# We pass our new generation to the decode function to check the Prior
# DECODING DNA
# CHECK PRIOR

# We add the immigration individuals
# MIGRATION