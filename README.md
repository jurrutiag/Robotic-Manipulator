# Robotic Manipulator

A robotic manipulator is a mechanism that is widely used in the industry, it consists of a robotic arm with joints and a hand or grabber. The main goal of the robotic manipulator is to reach a certain point in space in order to perform a certain action, this is done through motors that can adjust the joint angles. Calculating a position given a set of joint angles is easy and is mainly done by applying trigonometrics, but finding a set of angles that makes the arm reach a certain target is not an easy task because there are many of them that fulfill that condition. In order to find the optimal one (where optimality is to be defined later) a genetic algorithm approach is proposed and will be developed throughout the semester.

The robotic manipulator will be modeled using the Denavit-Hartenberg convention for selecting reference frames.

This project is developed in the context of the subject EL4106 - Inteligencia Computacional from Universidad de Chile. 

# How to use
The Model folder contains 'main.py', the file that can run the algorithm importing the relevant parts.

'RoboticManipulator.py' contains the logic of the manipulator itself, 'FitnessFunction.py' contains the logic of the fitnes function used, and 'GeneticAlgorithm.py' features the algorithm itself, using individuals modeled with the Individual class contained in 'Individual.py'.


## Authors
* Javier Mosnaim
* Juan Urrutia
