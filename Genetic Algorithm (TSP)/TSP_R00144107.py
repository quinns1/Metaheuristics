#!/usr/bin/env python

"""
Name: Shane Quinn
Student Number: R00144107
Email: shane.quinn1@mycit.ie
Course: MSc Artificial Intelligence
Module: Metaheuristic Optimization 
Date: 06/11/2020
"""

import random
from Individual import *
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt  
import time

myStudentNum = 144107                   #Student Numbe: R00144107
random.seed(myStudentNum)
# random.seed()
v=False

class TSP_R00144107:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations, _initPopulationType, _crossoverType, _mutationType):
        """
        Parameters and general variables
        """

        self.population     = []
        self.new_population = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = _popSize
        self.genSize        = None
        self.mutationRate   = _mutationRate
        self.maxIterations  = _maxIterations
        self.iteration      = 0
        self.fName          = _fName
        self.data           = {}
        self.initPopType    = _initPopulationType
        self.crossoverType  = _crossoverType
        self.mutationType   = _mutationType
        self.first          = True
        self.improvement    = []                                #Holds iteration-current best solution. Updated after each best is updated
        if v:
            print("\n----- Genetic Algorithm Configuration -----")
            print("Population: {}\nMutation Rate: {},\nMax Iterations: {}\nCrossover Type: {},\nMutation Type: {}\nInitial Population Type: {}".format(
                self.popSize, self.mutationRate, self.maxIterations, self.crossoverType, self.mutationType, self.initPopType))
            print("Input File: {}\n".format(self.fName))
        self.readInstance()
        self.initPopulation()
        


    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (cid, x, y) = line.split()
            self.data[int(cid)] = (int(x), int(y))
        file.close()


    def initPopulation(self):
        """
        Creating initial population of size self.popSize. Use either NearestNeighbour or Random approach
        """      

        for i in range(0, self.popSize):
            if self.initPopType == 'NearestNeighbour':
                individual = Individual(self.genSize, self.nearest_neighbour(), [])
            elif self.initPopType == 'Random':
                individual = Individual(self.genSize, self.rand_tour(), [])
            individual.computeFitness()
            self.population.append(individual)
            
        self.best = self.population[0].copy()
        for ind_i in self.population:
            
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        # print(self.best.fitness)
        self.improvement.append([1,self.best.getFitness()]) 
        self.stochasticUniversalSampling()
            
    def rand_tour(self):
        """
        Randomly selects individuals genes (random route)

        Returns
        -------
        rt : Dictionary
            Randomely selected individual genes.
        """
        
        rt = {}
        posns_hit = []
        val_list = list(self.data.values())
        key_list = list(self.data.keys())
        first_key = random.choice(key_list)                                             #generate random first key
        current_key = first_key
        rt[key_list[val_list.index(self.data[current_key])]] = self.data[current_key]  
        posns_hit.append(self.data[current_key])
        rand_posn = self.data[current_key]
        
        while len(posns_hit) < len(self.data):
            
            while rand_posn in posns_hit:    
                rand_key = random.choice(key_list)
                rand_posn = self.data[rand_key]     
                
            current_key = rand_key
            posns_hit.append(rand_posn)
            rt[key_list[val_list.index(self.data[current_key])]] = self.data[current_key] 
            
        return rt


    def nearest_neighbour(self):
        """
        Generates Individual genes using nearest neighbour heuristic
        Starting Position picked at random
        
        Returns
        -------
        nn : Dictionary
            Nearest Neighbour Individual Genes 
        """

        nn = {}                                                                             #Define dictionary {Location: (x-cordinate, y-cordinate)}    
        posns_hit = []
        val_list = list(self.data.values())                                                 #Save locations and names
        key_list = list(self.data.keys())
        first_key = random.choice(key_list)                                                 #generate random first key
        current_key = first_key
        nn[key_list[val_list.index(self.data[current_key])]] = self.data[current_key]       #Add first location to  dictionary
        posns_hit.append(self.data[current_key])                                            #Save locations already visited
        
        while len(posns_hit) < len(self.data):                                              #iterate through all locations, until all have been visited
            closest_posn, current_key = self.find_closest(self.data[current_key], posns_hit) #Find closest location to current location
            posns_hit.append(closest_posn)                                                  #Add closests tod dictionary and visited list
            nn[ key_list[val_list.index(self.data[current_key])]] = self.data[current_key]  
            
        return(nn)                                                                          #Return dictionary         

    
    def find_closest(self, node, ph):
        """
        Take in one position and locations already visited and return closest location that hasn't been visited

        Parameters
        ----------
        node : TYPE
            DESCRIPTION.
        ph : LIST
            POSITIONS HIT (LOCATIONS ALREADY VISITED).

        Returns
        -------
        lowest_posn : TUPLE
            CLOSEST POSITION.
        lowest_key : INT
            CLOSEST POSITION KEY.
        """
        
        lowest_weight = 999999999999999999999999     #impossible high distance
        lowest_key = None
        lowest_posn = None
        
        for i in self.data:
            temp_weight = self.distance(node, self.data[i])
            if temp_weight < lowest_weight and temp_weight !=0 and self.data[i] not in ph:
                    lowest_weight = temp_weight
                    lowest_posn = self.data[i]
                    lowest_key = i  
                    
        return lowest_posn, lowest_key

            
    def distance(self,vert_a, vert_b):
        """
        Calculates Euclidean distance between two supplied point     

        Parameters
        ----------
        vert_a : TUPLE
            X,Y CO ORDINATES OF POSITION 1.
        vert_b : TUPLE
            X,Y CO ORDINATES OF POSITION 2.

        Returns
        -------
        weight: INT
                DISTANCE BETWEEN TWO POINTS
        """
        
        weight = (vert_a[0]-vert_b[0])**2 + (vert_a[1]-vert_b[1])**2
        weight = int(round(math.sqrt(weight)))
        
        return weight


    def updateBest(self, candidate):
        """
        Take in individual, compare against current best. If better, update best

        Parameters
        ----------
        candidate : INDIVIDUAL OBJECT
            CANDIDATE BEST INDIVIDUAL.

        Returns
        -------
        None.

        """
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            # print ("iteration: ",self.iteration, "best: ",self.best.getFitness())
            self.improvement.append([self.iteration,self.best.getFitness()])
            
            
            
            
    def binaryTournamentSelection(self):
        """
        Randomly select 4 Individuals from mating pool. Compare IndA to IndB and IndC to IndD in 'tournament'. Winner of each tournament
        is returned.

        Returns
        -------
        winner1 : INDIVIDUAL OBJECT
            FITTER INDIVIDUAL OUT OF INDA AND INDB.
        winner2 : INDIVIDUAL OBJECT
            FITTER INDIVIDUAL OUT OF INDC AND INDD.

        """

        r1 = random.randint(0, self.popSize-1)                          #Select 4 random individuals from matting pool
        r2 = random.randint(0, self.popSize-1)
        r3 = random.randint(0, self.popSize-1)
        r4 = random.randint(0, self.popSize-1)
        indA = self.matingPool[ r1 ]
        indB = self.matingPool[ r2 ]
        indC = self.matingPool[ r3 ]
        indD = self.matingPool[ r4 ]
        winner1 = None
        winner2 = None
        if indA.getFitness() < indB.getFitness():                       #Compare fitenesses and return winners
            winner1 = indA
        else:
            winner1 = indB
            
        if indC.getFitness() < indD.getFitness():
            winner2 = indC
        else: 
            winner2 = indD
        
        return winner1, winner2

    def stochasticUniversalSampling(self):
        """
        Set the probability of each individual being selected from the population to be included in the mating pool
        in such a way that it reflects the Individuals fitness
        The fitter an individual is the more likely it is to be selected.
        
        See Accompanied Report for more details

        Returns
        -------
        None.

        """
        
        f = self.getTotalFitness()                      
        n = self.popSize
        p = 1/n                                     #total population is divided between 0 - 1
        rangeDict = {}                              #Dictionary where we will save each individuals upper and lower bounds
        point = random.uniform(0,p)                 #Random starting position, each selection point is + 1/populationsize from here (population size = amount of points)
        start = 0
        
        for i in self.population:                   #Iterate through population, fitnes
            fit = 1/i.fitness                       #
            line_segment = round(fit/f,4)                   #Fitness reflected line segment calculated here and round to 4 decimal places
            rangeDict[i] = (start, start+line_segment)      #Start - end for this individual saved in range dictionary
            start += line_segment                   #Move start to current end for next individual
            start = round(start,4)                  #Round start to 4 decimal places
            last = i 
                               
        x = rangeDict[last][0]                      #limit final range to below 1 
        rangeDict[last]=(x,1)
        
        for a in range(self.popSize):               #iterate through 'points' on line segment and add individuals to mating pool
                                                    #if point is within their range.
            for i in self.population:
                if point <= rangeDict[i][1]:
                    if point > rangeDict[i][0]:
                        self.matingPool.append(i)
                        point += p
                        break            
            
        
    def getTotalFitness(self):
        """
        Find inverse of distance because Stochastic Universal Sampling relies on larger values
        for higher fitness

        Returns
        -------
        tf : FLOAT
            INDIVIDUAL TOTAL FITNESS.

        """
        tf=0
        for i in self.population:
            i.computeFitness()
            tf += 1 / i.getFitness()                        
        return tf
    
     
    
    def uniformCrossover(self, indA, indB):
        """
        Inversion Crossover Implementation:
            2 Parents taken in, Child A is a copy of Parent A, Child B is a copy of Parent B
            Binary representation of length gene size is generated, the probability of each bit being 1 is
            set in by the change_prob variable.
            We iterate through parent A and B genes and whenever the corosponding bit in the binary representation = 1
            we remove this chromosome.
            The remaining chromosomes are appended onto the remaining chromosomes in the order that they appear in the other
            parent.        
            Children are returned
        
        See Accompanied Report for more details

        Parameters
        ----------
        indA : INDIVIDUAL OBJECT
            PARENT 1.
        indB : INDIVIDUAL OBJECT
            PARENT 2.

        Returns
        -------
        child1 : TYPE
            DESCRIPTION.
        child2 : TYPE
            DESCRIPTION.

        """
        cgenes1 = copy.deepcopy(indA.genes)                         #Copy made of Parents genes
        cgenes2 = copy.deepcopy(indB.genes)
        change_prob = .7
        binary_rep = []

        
        while len(binary_rep) < self.genSize:                       #Create binary representation of if gene index is removed
            if random.random() > change_prob:
                binary_rep.append(1)
            else:
                binary_rep.append(0) 

        for i in range(self.genSize):                               #Label chromosomes to be removed as 'gone'
            if binary_rep[i] == 1:
                cgenes1[i] = "gone"
                cgenes2[i] = "gone"
             
        for i in range(self.genSize):                               #Append chromosomes from Parent B onto child A and Parent A onto Child B
            if indB.genes[i] not in cgenes1:
                cgenes1.append(indB.genes[i])
            if indA.genes[i] not in cgenes2:
                cgenes2.append(indA.genes[i])
            try:                                                    #Remove chromosomes labelled 'gone'
                cgenes1.remove("gone")
                cgenes2.remove("gone")                
            except:
                pass       

        child1 = Individual(self.genSize, self.data, cgenes1)       #Return Children
        child2 = Individual(self.genSize, self.data, cgenes2)
        
        return child1, child2
        
        

    def order1Crossover(self, indA, indB):
        """
        Order-1 Crossover Implementation:
            2 Parents taken in, Child A is a copy of Parent A, Child B is a copy of Parent B
            chromosones are removed from random start to end point of each,
            removed chromosones from Child A are appended onto Child B
            Removed chromosones from Child B are appended onto Child A
            Children are returned
        
        See Accompanied Report for more details

        Parameters
        ----------
        indA : INDIVIDUAL OBJECT
            PARENT 1.
        indB : INDIVIDUAL OBJECT
            PARENT 2.

        Returns
        -------
        child1 : TYPE
            DESCRIPTION.
        child2 : TYPE
            DESCRIPTION.

        """
        
        cgenes1 = copy.deepcopy(indA.genes)                                     #Copy of Parent Genes
        cgenes2 = copy.deepcopy(indB.genes)       
        start_index = random.randint(0, len(indA.genes)-2)                      #Pick a random start index
        end_index = random.randint(start_index, len(indA.genes)-1)              #And a random end index
        rm_chromosones1 = []
        rm_chromosones2 = []                                                    #Define two lists for keeping track of chromosones to remove
        
        
        for i in range(start_index, end_index+1):                               #Fill with chromosones to be removed from parent1 and 2 
            rm_chromosones1.append(cgenes1[i])
            rm_chromosones2.append(cgenes2[i])
        
        
        for i in rm_chromosones1:                                                                                          
            cgenes2.remove(i)                                                   #Remove chromosones
            
        for i in rm_chromosones2:
            cgenes1.remove(i)
            
        for i in range(self.genSize):
            if indA.genes[i] not in cgenes2:                                    #Add chromosones removed from Parent A to Child B and Parent B - Child A
                cgenes2.append(indA.genes[i])
            if indB.genes[i] not in cgenes1:
                cgenes1.append(indB.genes[i])
                
        child1 = Individual(self.genSize, self.data, cgenes1)                   #Define children as individuals
        child2 = Individual(self.genSize, self.data, cgenes2)
        
        
        return child1, child2                                                   #Return children
    
    def scrambleMutation(self, ind):
        """
        Scramble Mutation Implementation:
            An individual is passed in which has a probability of self.mutationRate of being mutated
            The mutation is achieved by randomly selecting a range of chromosones and shuffling them
            
        See Accompanied Report for more details

        Parameters
        ----------
        ind : INDIVIDUAL OBJECT
            INDIVIDUAL TO BE MUTATED.

        Returns
        -------
        None.

        """
        
        if random.random() > self.mutationRate:                                 #Select random number between 0-1
            return                                                              #If numbe is greater than mutation rate, don't mutate
        
        start_index = random.randint(0, len(ind.genes)-2)
        end_index = random.randint(start_index, len(ind.genes)-1)               #Get random range start-end  
        temp = ind.genes
        sliced = temp[start_index:end_index+1]                                  #Create copy of chromosomes in this range
        random.shuffle(sliced)                                                  #Shuffle selected chromosomes
        index = start_index
        for i in sliced:                                                        #Re-insert shuffled chromosomes into genes
            temp[index] = i
            index +=1
        ind.genes = temp

        
        
    def inversionMutation(self, ind):
        """
        Inversion Mutatin Implementation:
            An individual is passed in which has a probability of self.mutationRate of being mutated
            The mutation is achieved by randomly selecting a range of chromosomes and inverting them

        See Accompanied Report for more details
        
        Parameters
        ----------
        ind : INDIVIDUAL OBJECT
            INDIVIDUAL TO BE MUTATED.

        Returns
        -------
        None.

        """
        
        if random.random() > self.mutationRate:                                 #Select random number if it's greater than mutation rate don't mutate
            return
        
        start_index = random.randint(0, len(ind.genes)-2)                       #Get random start and end index
        end_index = random.randint(start_index, len(ind.genes)-1)  
        temp = ind.genes                                                        
        sliced = temp[start_index:end_index+1]                                  #Create slice of chromosomes from start - end index        
        sliced.reverse()                                                        #Reverse chromosomes
        index = start_index
        
        for i in sliced:
            temp[index] = i                                                     #Reinsert reversed chromosomes into individual genes
            index +=1
            
        ind.genes = temp
    
        
    
    def crossover(self, indA, indB):
        """
        Executes a dummy crossover and returns the genes for a new individual
        """
        print("here")
        midP=int(self.genSize/2)
        cgenes = indA.genes[0:midP]
        for i in range(0, self.genSize):
            if indB.genes[i] not in cgenes:
                cgenes.append(indB.genes[i])
        child = Individual(self.genSize, self.data, cgenes)
        return child

    def mutation(self, ind):
        """
        Mutate an individual by swaping two cities with certain probability (i.e., mutation rate)
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp
        
        

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        self.matingPool = []
        if self.first:                                      #If first iteration call proportional selection otherwise add children to population first
            self.stochasticUniversalSampling()
            self.first=False   
        else:            
            self.population = []
            
            for i in self.new_population:                   #Update population with offspring
                self.population.append(i)
                
            self.stochasticUniversalSampling()              #Proportional selection
            self.new_population=[]                          #empty new population, children added here
            

    def newGeneration(self):
        """
        Create a new generation

        Returns
        -------
        None.

        """
        for i in range(0, int(len(self.population)/2)):            
            parent1, parent2 = self.binaryTournamentSelection()
            if self.crossoverType == 'Uniform':
                child1, child2 = self.uniformCrossover(parent1, parent2)
            elif self.crossoverType == 'Order-1':
                child1, child2 = self.order1Crossover(parent1, parent2)
            else:
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent1, parent2)
            
            if self.mutationType == 'Inversion':
                self.inversionMutation(child1)
                self.inversionMutation(child2)
            elif self.mutationType == 'Scramble':
                self.scrambleMutation(child1)
                self.scrambleMutation(child2)
            else:
                self.mutation(child1)
                self.mutation(child2)
            
            child1.computeFitness()
            child2.computeFitness()
            self.updateBest(child1)
            self.updateBest(child2)

            self.new_population.append(child1)
            self.new_population.append(child2)


    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """

        self.updateMatingPool()
        self.newGeneration()

    def search(self):
        """
        General search template.
        Iterates for a given number of steps
        """
        
        self.iteration = 0
        
        while self.iteration < self.maxIterations:
            self.GAStep()
            self.iteration += 1
            
        if v:
            print ("Total iterations: ", self.iteration)
            print ("Best Solution: ", self.best.getFitness())
        self.improvement.append([self.iteration,self.best.getFitness()])
        val_list = list(self.best.data.values())
        key_list = list(self.best.data.keys())

        
        return self.improvement             
    
        

def runTSP(configx):
    """
    Give in configuration details to run tsp. Configuration includes:
        - Config name                           = Configuration Name
        - Problem File                          = Input file
        - Population Size                       = Population Size
        - Mutation Rate                         = Probability of each new individual being mutated 
        - Max Iterations                        = Number of iterations
        - Initial Population Generation Type    = Random/NearestNeighbour
        - Crossover Type                        = Order-1/Uniform
        - Mutation Type                         = Inversion/Scramble
        - Runs                                  = Numbe of runs

    Passes these parameters to TSP GA program. Which returns dict of {iteration: best} (this is updated every time best is updated)
    Improvments are plotted using matplotlib.
    Mean execution time and mean best is printed in console
    Parameters
    ----------
    configx : LIST
        DETAILED ABOVE.

    Returns
    -------
    None.

    """
    
    cf, pf, ps, mr, mi, ipt, c, m, r = configx  
    results=[]
    execution_time = []
    best = []  
    
    for i in range(r):
        start_time = time.time()
        ga = TSP_R00144107(pf, ps, mr, mi, ipt, c, m)
        results.append(ga.search())
        execution_time.append(time.time()-start_time)
            
    for i in range(len(results)):
        best.append(results[i][len(results[i])-1][-1])
        
    mean_best = sum(best)/len(best)
    mean_time = sum(execution_time)/len(execution_time)    
    print("Config: {}\nBest Results: {}\nExecution Times: {}\nMean Best Results: {}\nMean Execution Time: {}s".format(
        cf, best, execution_time, mean_best, mean_time))    
    results = np.array(results)
    x=[]

    
    for i in range(len(results)):                               #Generate Plots
        x.append(np.array(results[i]))
        plt.plot(x[i][:,0], x[i][:,1], color ="blue", zorder=i)  
    

    plt.title(cf)  
    plt.xlabel("Iterations")  
    plt.ylabel("Distance")  
    plt.show()



def run_all_input_files():  
    """
    Run all relevant problem files ('inst-4.tsp','inst-6.tsp', 'inst-16.tsp')

    Returns
    -------
    None.

    """
    
    popsize, mutationRate, maxit, runs = 100, .05, 500, 5     #Population Size, Mutation Rate, Max Iterations, Number of Runs
    input_files_list = ['inst-4.tsp','inst-6.tsp', 'inst-16.tsp']
    for i in input_files_list:
        problem_file = i
        config1 = ['Config 1', problem_file, popsize, mutationRate, maxit, 'Random', 'Order-1', 'Inversion', runs]
        config2 = ['Config 2', problem_file, popsize, mutationRate, maxit, 'Random', 'Uniform', 'Scramble', runs]
        config3 = ['Config 3', problem_file, popsize, mutationRate, maxit, 'Random', 'Order-1', 'Scramble', runs]
        config4 = ['Config 4', problem_file, popsize, mutationRate, maxit, 'Random', 'Uniform', 'Inversion', runs]
        config5 = ['Config 5', problem_file, popsize, mutationRate, maxit, 'NearestNeighbour', 'Order-1', 'Scramble', runs]
        config6 = ['Config 6', problem_file, popsize, mutationRate, maxit, 'NearestNeighbour', 'Uniform', 'Inversion', runs]
        configs = [config1, config2, config3, config4, config5, config6] 
        
        # configs = [config2, config4, config6]
        for con in configs:
            runTSP(con)
            print(con)

def run_single_file(p):
    """
    Run single input file. Runs all configurations of TSP GA on problem file

    Parameters
    ----------
    p : FILE
        INPUT FILE.

    Returns
    -------
    None.

    """
    
    popsize, mutationRate, maxit, runs = 100, .05, 500, 5      #Population Size, Mutation Rate, Max Iterations, Number of Runs
    
    problem_file = p   
    config1 = ['Config 1', problem_file, popsize, mutationRate, maxit, 'Random', 'Order-1', 'Inversion', runs]
    config2 = ['Config 2', problem_file, popsize, mutationRate, maxit, 'Random', 'Uniform', 'Scramble', runs]
    config3 = ['Config 3', problem_file, popsize, mutationRate, maxit, 'Random', 'Order-1', 'Scramble', runs]
    config4 = ['Config 4', problem_file, popsize, mutationRate, maxit, 'Random', 'Uniform', 'Inversion', runs]
    config5 = ['Config 5', problem_file, popsize, mutationRate, maxit, 'NearestNeighbour', 'Order-1', 'Scramble', runs]
    config6 = ['Config 6', problem_file, popsize, mutationRate, maxit, 'NearestNeighbour', 'Uniform', 'Inversion', runs]
    configs = [config1, config2, config3, config4, config5, config6]  
    # configs = [config3, config4, config5, config6]
    for con in configs:
        runTSP(con)
        print(con)



def run_exp_config(p):
    """
    Run single input file on specific configuraiton. Alter config below

    Parameters
    ----------
    p : FILE
        INPUT FILE.

    Returns
    -------
    None.

    """
    
    problem_file = p
    popsize, mutationRate, maxit, runs = 100, .05, 500, 5       #Population Size, Mutation Rate, Max Iterations, Number of Runs
    config = 'Config X'                                         #Config Name
    initPopType = 'NearestNeighbour'                            #'Random'/'NearestNeighbour'
    crossoverType = 'Uniform'                                   #'Order-1'/'Uniform'
    mut = 'Inversion'                                           #'Inversion'/'Scramble'
    configx = [config, problem_file, popsize, mutationRate, maxit, initPopType, crossoverType, mut, runs]
    runTSP(configx)
    
    
    
def main():
    """
    Runs chosen function.    
    
    COMMENT OUT FUNCTIONS NOT BEING RUN

    Returns
    -------
    None.

    """

    # run_single_file(problem_file)
    run_exp_config(problem_file)
    # run_all_input_files()
    
    
    
    
   
try:
    problem_file = sys.argv[1]
    run_single_file(problem_file)
except:
    problem_file = 'inst-4.tsp'    

if __name__=='__main__':
    main()
    