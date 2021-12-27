# -*- coding: utf-8 -*-
"""
Name: Shane Quinn
Student Number: R00144107
Email: shane.quinn1@mycit.ie
Course: MSc Artificial Intelligence
Module: Metaheuristic Optimization 
Date: 14/12/2020
"""

import random
import sys
import copy
import matplotlib.pyplot as plt  
import time
import math

myStudentNum = 144107                   #Student Numbe: R00144107
random.seed(myStudentNum)


class TSP_2_Opt:
    
    def __init__(self, fName, variant, mult):
        
        self.file = fName
        self.m = mult
        self.nodes_count = 0
        self.data = {}
        self.var = variant
        self.readInstance()
        self.tour, self.edges = self.nearest_neighbour()
        self.current_cost = sum(self.edges.values())
        # print("Cost after initial solution: ", self.current_cost)   
        self.run2Opt()
        

    def run2Opt(self):
        """
        Run variation of 2-opt specified by self.var
        either of the following are valid options:
            - 'basic'
            - 'var1'
            - 'var2'

        Returns
        -------
        None.

        """
        if self.var == 'basic':
            self.run_basic()
        elif self.var == 'var1':
            self.run_var1()
        elif self.var == 'var2':
            self.run_var2()
            
    
    def run_basic(self):
        """
        Basic 2-Opt Local Search 
        Iterates through all edge swaps, records best for each outer iteration (or first edge) and executes it after checking each inner iteration or second edge
        Edges are denoted by their first city so for tour T = [a, b, c, d], edge index 2 = b, which signifies the edge [b - c]
        All edges are denoted by index in current tour.

        Returns
        -------
        None.

        """
        notOpt = True
        while notOpt == True:                       #Repeat until reaches optimal solution
            notOpt = False
            i = 0
            n = self.nodes_count                    #Number of cities in tour
            currBest = self.current_cost            #Current tour cost
            while i < n - 3:                        #Iterate through all possible first edges (all edges excluding last edge or last 2 cities)
                j = i + 2                           #Possible swap edges start 2 after current edge
                while j < n-2:                      #Iterate through all possible 2nd edges
                    D = self.swap_cost(i, j)        #Calculate the cost of entire tour after swaping 2 edges        
                    if D < currBest:                #If cost is less than current best, save swap as best move
                        currBest = D
                        bestMove = [i, j]                        
                        notOpt = True
                    j += 1
                i += 1
            if notOpt == True:                      #If there is an optimal solution update tour
                self.update_tour(bestMove)
    
    
    def run_var1(self):
        """
        Variation of previous 2-Opt function.
        Picks random edge, search all possible swaps with edge. Execute best swap as before
        iteration limit set by (number of cities) * self.m
        self.m is defined by the user
        
        Returns
        -------
        None.

        """
        notOpt = True
        iterations = 0
        while iterations < self.nodes_count*self.m:         #Iteration limit set by user (number of cities) * self.m
            notOpt = False
            n = self.nodes_count
            i = random.randint(0, n-4)       #Pick random first city index
            currBest = self.current_cost 
            j = 0                               #j = first city index of swap edge
            while j < n-2:                      #Iterate through all possible swap edges
                if j == i or j == i+ 1:         #Excluding edge we're currently searching fo
                    j += 1
                    continue
                D = self.swap_cost(i, j)        #Calculate the cost of swapping 2 edges
                if D < currBest:                #If better than current best save
                    currBest = D
                    bestMove = [i, j]                        
                    notOpt = True
                j += 1
            iterations += 1
            if notOpt == True:                  #If there is an optimal solution update tour
                self.update_tour(bestMove)

    def run_var2(self):
        """
        This is a variation which derives from var1 above. In this variation, isntead of checking every edge for the best
        solution. We execute the first swap which gives a better solution to the current sol. Again we have an iteration limit
        set by (number of cities) * self.m

        Returns
        -------
        None.

        """
        notOpt = True
        iterations = 0
        while iterations < self.nodes_count*self.m:    #Iteration Limit
            notOpt = False
            n = self.nodes_count                
            i = random.randint(0, n-4)                  #Pick random edge
            currBest = self.current_cost
            j = 0                                   #Iterate through all possible swap edges
            while j < n-2:
                if j == i or j == i+1:
                    j += 1
                    continue
                D = self.swap_cost(i, j)            #Calculate tour cost after swap.
                if D < currBest:                    #If better than current, update tour
                    currBest = D
                    bestMove = [i, j]                        
                    notOpt = True
                if notOpt == True:
                    break
                j += 1
            iterations += 1
            if notOpt == True:
                self.update_tour(bestMove)
   
        
        
    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.file, 'r')
        self.nodes_count = int(file.readline())
        self.data = {}
        for line in file:
            (cid, x, y) = line.split()
            self.data[int(cid)] = (int(x), int(y))
        file.close()
        
        
    def update_tour(self, best):
        """
        Take in edge indexes and swap.
        Empty tour and edges dictionaries
        iterate up to start of first edge and fill dictionaries accordingly
        iterate from start of second edge to end of first edge and fill dicts accordingly
        iteraate from end of second edge to end of tour and fill dicts accordingly
        
        Tour = [1,2,3,4,5,6,7,8]   - If we wish to swap edges [2,3] and [6,7] the resulting tour would be as follows
        Tour = [1,2,6,5,4,3,2,7,8]

        Parameters
        ----------
        best : LIST
            Idexes of edges to be swapped [i, k].
            i = edge1 first index
            j = edge2 first index

        Returns
        -------
        None.

        """
        
        
        i = best[0]         #edge1 index, (edge1 second index = i + 1)
        k = best[1]         #edge2 index
        nt = {}             #new tour dictionary
        val_list = list(self.tour.values())     #Current tour keys 
        key_list = list(self.tour.keys())       #Current tour values
        edges = {}              #New edges dictionary
        x = 0                   #new tour index

        while x <= i:               #iterate up to first edge
            k1 = key_list[x]        
            nt[k1] = val_list[x]    #Add city to tour dictionary
            if x >= 1:
                edge_key = (k2, k1)         #Edges are identified by city keys on either side of edge
                edges[edge_key] = self.distance(nt[k2], nt[k1])  #Add edge to edges dictionary   
            x += 1
            k2 = k1
        x = k           #Skip to start of second edge and work backwards to end of first edge
        while x > i: 
            k1 = key_list[x]
            nt[k1] = val_list[x]   
            edge_key = (k2, k1)
            edges[edge_key] = self.distance(nt[k2], nt[k1])
            x -= 1  
            k2 = k1
        x = k + 1       #Skip to end of second edge and work to end
        while x < len(key_list):
            k1 = key_list[x]
            nt[k1] = val_list[x]
            edge_key = (k2, k1)
            edges[edge_key] = self.distance(nt[k2], nt[k1])
            x += 1
            k2 = k1
        
        
        self.tour = nt          #Update class tour dict
        self.edges = edges      #Update class edges dict
        edge_key = (key_list[-1], key_list[0])
        self.edges[edge_key] = self.distance(self.data[key_list[-1]], self.data[key_list[0]])   #Append trip to first city to edges
        self.current_cost = sum(self.edges.values())            #Calculate new current cost


        
        
    def swap_cost(self, e1, e2):
        """
        Calculate the cost of swapping 2 edges
        Cost = Original Cost - Old Edge Costs + New Edge Costs

        Parameters
        ----------
        e1 : INT
            Index indicating edge 1 first point in current tour.
        e2 : INT
            Index indicating edge 2 first point in current tour.

        Returns
        -------
        cost : INT
            Calculated cost of swapping edges.

        """
        
        key_list = list(self.tour.keys())    
        cost = sum(self.edges.values())     
        edge1_a = key_list[e1]
        edge1_b = key_list[e1 + 1]
        edge2_a = key_list[e2]
        edge2_b = key_list[e2 + 1]     
       
        drop_edges_cost = self.edges[(edge1_a, edge1_b)] + self.edges[edge2_a, edge2_b]
        new_edge_cost = self.distance(self.tour[edge1_a], self.tour[edge2_a]) + self.distance(self.tour[edge1_b], self.tour[edge2_b])
        cost -= drop_edges_cost
        cost += new_edge_cost   
        
        return cost
    





    def nearest_neighbour(self):
        """
        Generates Individual genes using nearest neighbour heuristic
        Starting Position picked at random
        
        Returns
        -------
        nn : Dictionary
            Nearest Neighbour Dictionary
        """

        nn = {}                                                                            #Define dictionary {Location: (x-cordinate, y-cordinate)}    
        edges = {}
        posns_hit = []
        val_list = list(self.data.values())                                                 #Save locations and names
        key_list = list(self.data.keys())
        first_key = random.choice(key_list)                                                 #generate random first key
        current_key = first_key
        nn[key_list[val_list.index(self.data[current_key])]] = self.data[current_key]       #Add first location to  dictionary
        posns_hit.append(self.data[current_key])                                            #Save locations already visited
        
        while len(posns_hit) < len(self.data):                                              #iterate through all locations, until all have been visited
            closest_posn, new_key, distance = self.find_closest(self.data[current_key], posns_hit) #Find closest location to current location
            edge_key = (current_key, new_key)
            current_key = new_key
            edges[edge_key] = distance
            posns_hit.append(closest_posn)                                                  #Add closests to dictionary and visited list
            nn[ key_list[val_list.index(self.data[current_key])]] = self.data[current_key] 
        
        edge_key = (current_key, first_key)
        edges[edge_key] = self.distance(self.data[current_key], self.data[key_list[0]])

        return nn, edges                                                                           #Return dictionary         


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
                    
        return lowest_posn, lowest_key, lowest_weight

            
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
        
        
        
      
        

def run_2_opt(p, variation = 'basic', iterations=5, mult = 5):
    """
    Run 2-opt algorithm on TSP file 'p' using variation specified. 
    Repeat 'iterations' number of times
    return mean

    Parameters
    ----------
    p : TSP File
        TSP text doc.
    variation : STRING, optional
        basic, var1, var2. The default is 'basic'.
    iterations : INT, optional
        Number of repeats. The default is 5.
    mult : INT, optional
        Multiple of number of cities (sets iteration limit in var1 and var2). The default is 5.

    Returns
    -------
    mean : INT
        Mean average across all iterations.

    """
    problem_file = p
    results = []
    execution_times = []
    for i in range(iterations):
        start_time = time.time()
        two_opt = TSP_2_Opt(problem_file, variation, mult)
        execution_times.append(time.time()-start_time)
        results.append(two_opt.current_cost)
        # print(results)
    mean = sum(results)/len(results)
    print("Problem File: ", p)
    print("Variation: ", variation)
    print("Results: ", results)
    print("Execution Times: ", execution_times)
    print("Mean Results: ", sum(results)/len(results))
    print("Mean Execution Times: ", sum(execution_times)/len(execution_times))
    return mean
    
    
def run_2_opt_all_vars(p, iterations=5, mult = 5):
    """
    Run all variations of 2 opt on problem file specified by 'p' for 'iterations' number of times.

    Parameters
    ----------
    p : TSP File
        TSP text doc.
    variation : STRING, optional
        basic, var1, var2. The default is 'basic'.
    iterations : INT, optional
        Number of repeats. The default is 5.
    mult : INT, optional
        Multiple of number of cities (sets iteration limit in var1 and var2). The default is 5.

    Returns
    -------
    mean : INT
        Mean average across all iterations.

    """
    variation_list = ['basic', 'var1', 'var2'] 
    problem_file = p
    for variation in variation_list:
        results = []
        execution_times = []
        for i in range(iterations):
            start_time = time.time()
            two_opt = TSP_2_Opt(problem_file, variation, mult)
            execution_times.append(time.time()-start_time)
            results.append(two_opt.current_cost)
            # print(results)
        mean = sum(results)/len(results)
        print("Problem File: ", p)
        print("Variation: ", variation)
        print("Results: ", results)
        print("Execution Times: ", execution_times)
        print("Mean Results: ", sum(results)/len(results))
        print("Mean Execution Times: ", sum(execution_times)/len(execution_times))
        print("\n\n")

        
       
        
def main():
    """
    Runs chosen function.    
    
    COMMENT OUT FUNCTIONS NOT BEING RUN

    Returns
    -------
    None.

    """
    
    problem_file = 'inst-16.tsp'
    iterations = 5   
    var = 'var1'
    
    'Uncomment below to run config specified above'
    run_2_opt(problem_file, var, iterations)
    
    'Uncomment below to run all variations of 2_opt on problem file specified below'
    # run_2_opt_all_vars(problem_file, iterations)
    
    'Uncomment below to investigate effect of changing iterations limit'
    # mults = [1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 50, 75]
    # res = []
    # for i in mults:
    #     res.append(run_2_opt(problem_file, var, iterations, mult=i))
    # plt.plot(mults, res)
    # plt.show()    
    

   
try:
    problem_file = sys.argv[1]
except:
    problem_file = 'inst-4.tsp'    



if __name__=='__main__':
    main()