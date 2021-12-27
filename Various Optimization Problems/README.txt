-----------------
PROJECT CONTENTS
-----------------
Two_Opt.py 
HillClimbing.py
Queens.py
Quinn_Shane_R00144107_MH2.doc

------------------------------------------------------------
TRAVELLING SALESMAN PROBLEM - 2-OPT SOLUTION WITH VARIATIONS
------------------------------------------------------------
--------------
Description
--------------

Aim of the project is to implement a 2-Opt algorithm to solve the travelling salesman problem (TSP). TSP can be described as the following:

"If a traveling salesman wishes to visit exactly once each of a list m cities (where the cost of traveling from city i to city j is cij) and then return to the home city, what is the least costly route the traveling salesman can take"

The methodologies implemented in this project are listed here:
- Basic 2-Opt
- Variation 1 
- Variation 2


------------------
Technologies Used
------------------
1. Project is implemented in Python 3.8
2. Used Spyder IDE 
3. Following Python packages were used:
	- random
	- sys
	- time
 	- copy
	- math
4. Non python native packages:
	- matplotlib.pyplot

-----------------------
Description of Programs
-----------------------

1. Two_Opt.py

- Basic 2-opt - Description in accompanied report

- Variation 1 - Description in accompanied report

- Variation 2 - Description in accompanied report


Each function is fully commented and should be viewed for further information.

----------------------------------
Instructions for Running/Compiling
----------------------------------
1. Ensure relavant packages are installed, as well as Python 3
2. Recommended use Spyder IDE as all relevant packages are pre-installed
3. Ensure relevant .tsp files are in same directory as both .py files Two_Opt.py)
4. Open Two_Opt.py in Spyder IDE, see the following locations and make alterations as need be:
	- main(): here we choose to run the following (COMMENT OUT FUNCTIONS NOT BEING RUN)
    		- run_2_opt() - run 2-opt algorithm on tsp on provided or default configuration. 
    		- run_2_opt_all_vars()- Run all variations of 2-opt algorithm
5. Can specify problem_file on line 496 or have the option of calling program with problem file "python Two_Opt.py inst-4.tsp"

-------------------------------
NQUEENS - RUNTIME DISTRIBUTION
-------------------------------.

--------------
Description
--------------

N-Queens is a popular search problem, if we have N queens on an N*N chessboard. How can we arrange the queens so that they are not attacking each other? 

------------------
Technologies Used
------------------
1. Project is implemented in Python 3.8
2. Used Spyder IDE 
3. Following Python packages were used:
	- random
	- sys
	- time
 	- copy
	- math
4. Non python native packages:
	- matplotlib.pyplot


-----------------------
Description of Programs
-----------------------

1. Queens.py
	- Includes NQueens problem class

2. HillClimbing.py
	- Implementation of MinMax hillcimbing algorithm for solving NQueens


Each function is fully commented and should be viewed for further information.

----------------------------------
Instructions for Running/Compiling
----------------------------------
1. Ensure relavant packages are installed, as well as Python 3
2. Recommended use Spyder IDE as all relevant packages are pre-installed
3. Ensure Queens.py is in the same directory as HillClimbing.py
4. Run HillClimbing.py and view results through console.
5. To include sideways move uncommentt line 71