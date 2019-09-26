import math
import numpy as np
import random
import Individual

def initialization(populationSize, initialAngles, finalAngles, armPivots, steps):
    P = np.zeros((populationSize, armPivots))
    results = []
    for ind in range(populationSize):
        for h in range(armPivots):
            #solo el extremo inicial esta fijo
            average = random.randrange(2, populationSize) 
            std = random.randrange(1,populationSize/6)
            R= abs(initialAngles[h]-finalAngles[h])
            
            for i in range(populationSize):
                #no estoy seguro si habra que poner step distinto
                noise = random.randrange(-3*R, 3*R)*math.exp(-(i-average)**2/(2*std**2))       
                P[i,h] = initialAngles[h] + (i-1)*(initialAngles[h]*finalAngles[h])/(populationSize-1)+noise


        results.append(P)

    #lista de individuos
    return results


