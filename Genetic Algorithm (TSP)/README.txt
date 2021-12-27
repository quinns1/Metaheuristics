--------------------------------------------------------
TRAVELLING SALESMAN PROBLEM - GENETIC ALGORITHM SOLUTION
--------------------------------------------------------

--------------
Introduction
--------------

Aim of the project is to implement a genetic algorithm to solve the travelling salesman problem (TSP). TSP can be described as the following:

"If a traveling salesman wishes to visit exactly once each of a list m cities (where the cost of traveling from city i to city j is cij) and then return to the home city, what is the least costly route the traveling salesman can take"

The methodologies implemented in this project are listed here:
- Order-1 Crossover
- Uniform Crossover
- Inversion Mutation
- Scramble Mutation
- Nearest Neighbour Heuristic
- Stochastic Universal Sampling
- Binary Tournament Selection

------------------
Technologies Used
------------------
1. Project is implemented in Python 3.8
2. Used Spyder IDE to compile
3. Following Python packages were used:
	- random
	- sys
	- time
 	- copy
	- math
4. Non python native packages:
	- numpy
	- matplotlib.pyplot

-----------------------
Description of Programs
-----------------------

1. Individual.py

This contains a class which is instantiated for each individual. Each individual has the following traits:
- genes: Locations in order (route)
- genSize: Number of chromosomes or locations on route
- fitness: Total distance to reach each chromosome/location and return to starting position
- data: Physical x/y coordinates of locations

2. TSP_R00144107.py
This is the main program which contains the following functions:
- readInstance - Read in problem file to self.data dictionary
- initPopulation -  Generate Initial Population
- rand_tour - Generate initial population randomely
- nearest neighbour - Generate initial population using nearest neighbour heuristic
- updateBest - check is new individual fitness best
- binarytournamentSelection - pick 4 individuals, return 2 fittest
- stochasticUniversalSampling - proportional selection
- getTotalFitness - total fiteness of entire population
- uniformCrossover - Crossover Operater 
- order1Crossover - Crossover Operater
- scrambleMutation - Mutation operator
- inversionMutation - Mutation Operator
- updateMatingPool - Updates mating pool from current population
- newGeneration - Create new generation from current mating pool
- GAStep - One iteration of GA
- search - repeat GAStep x number of iterations
- runTSP - Take in config details, run TSP
- run_all_input_files - Run inst-4, inst-6, inst-16 (all configs)
- run_single_file - Run provided problem file (all configs)
- run_exp_config - Alter configuration and run modified, for experimentation


Each function is fully commented and should be viewed for further information.

----------------------------------
Instructions for Running/Compiling
----------------------------------
1. Ensure relavant packages are installed, as well as Python 3
2. Recommended use Spyder IDE as all relevant packages are pre-installed
3. Ensure relevant .tsp files are in same directory as both .py files (TSP_R100144107 and Individual.py)
4. Open TSP_R00144107.py in Spyder IDE, see the following locations and make alterations as need be:
	- main(): here we choose to run the following (COMMENT OUT FUNCTIONS NOT BEING RUN)
    		- run_single_file(problem_file) - Pass in .tsp file to run all 6 pre-configured configs on
    		- run_exp_config(problem_file) - Pass in .tsp file to run user defined configuration (MAKE CHANGES TO THIS FUNCTION AS REQUIRED)
    		- run_all_input_files() - Run all relevant .tsp files (in this case inst-4.tsp, inst-6.tsp, inst-16.tsp)
5. Can specify problem_file on line 810 or have the option of calling program with problem file "python TSP400177107.py inst-4.tsp"


