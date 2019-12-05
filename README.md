# Robotic Manipulator

A robotic manipulator is a mechanism that is widely used in the industry, it consists of a robotic arm with joints and a hand or grabber. The main goal of the robotic manipulator is to reach a certain point in space in order to perform a certain action, this is done through motors that can adjust the joint angles. Calculating a position given a set of joint angles is easy and is mainly done by applying trigonometrics, but finding a set of angles that makes the arm reach a certain target is not an easy task because there are many of them that fulfill that condition. In order to find the optimal one (where optimality is to be defined later) a genetic algorithm approach is proposed and will be developed throughout the semester.

The robotic manipulator will be modeled using the Denavit-Hartenberg convention for selecting reference frames.

This project is developed in the context of the subject EL4106 - Inteligencia Computacional from Universidad de Chile. 

# How to use
The root directory contains 'main.py', the file that can run the algorithm importing the relevant parts. The following modules are called inside main.py:

* **RoboticManipulator.py:** Module containing the logic of the robotic manipulator, this defines the manipulator object which contains the joint masses, length of the arms and limits of the motors.

* **FitnessFunction.py:** Module containing the logic of the fitness function, calculations of distance, torque and velocity are done with the FitnessFunction object.

* **GeneticAlgorithm.py:** Module that contains the GeneticAlgorithm class, this contains all the logic of the continuous genetic algorithm used for solving the optimization problem.

* **Individual.py:** Contains the class for the representation of an individual.

* **DisplayHandler.py:** This module is in charge of displaying the information to the UI or terminal.

* **JSONSaveLoad.py:** This module contains the class that handles the load and creation of .json files for the algorithm input and output.

* **MultiCoreExecuter.py:** This module is in charge of executing all the runs of the algorithm.

* **PrintModule.py:** Module used to print the output in a readable way.

# Pre installations

Install virtual environment from pip and then create a virtual environment called "env" on the project folder and activate it using the following commands:

```
py -m pip install --user virtualenv
py -m venv env
.\env\Scripts\activate
```

Install, with the virtual environment activated, the file requirements.txt with the following command from the root folder:

```
pip install -r requirements.txt
```

Then, in order to be able to visualize the data with the UI, please run:
```
garden install matplotlib
```
On windows the previous command throws an error, this is because the path to garden.py may have spaces in between, in order to fix this, copy the command that appears as ran on the error thrown, and type double quotes around the path, for example if the error throws that this command was called:

```
python.exe Path\To\gar
den install matplotlib
```

Rewrite as the following command and run it:

```
python.exe "Path\To\gar
den" install matplotlib
```

## Authors
* Javier Mosnaim
* Juan Urrutia
