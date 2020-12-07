# Multivariate-Hidden-Markov-Model-for-Electricity-Cost-Reduction
The problem considered is that of minimizing electricity costs in a building. Using an extension of a Hidden Markov chain to the multivariate case, we give one solution to this problem. The model decides whether to turn each light in a room on or off in an attempt to minimize the total electricity cost in one day's simulation, subject to penalties for leaving a light on in an empty room or off in an occupied room. People in the building cannot be directly observed. The model only has access to the outputs of certain sensors, which may or may not be reliable, may or may not fail, and cannot see every room.

This problem was the result of a university assignment - Assignment 2, COMP9418 Advanced Statistical Methods in Machine Learning, UNSW Sydney, Term 3 2020.

See the COMP9418-Assignment-2.pdf for the problem statement. 
See report.pdf for the solution outline, including a full explanation of the model and derivation of the mathematics.

Code includes all the code used to solve the problem. Includes a building simulator for testing. 
Parameters contains all the problem parameters (as csv files) that are loaded by the code in creating the solution. 

Alternatively, the included jupyter notebooks give an interactive demonstration of the code (one for learning the parameters, one for prediction and testing).
