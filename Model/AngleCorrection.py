import math
import numpy as np
import random
import Individual
def angleCorrection(minAngles, maxAngles, individuals):

    for ind in individuals:
        for i in individuals.shape(0):
            for h in shape(1):
                dif = abs(individuals[i,h]-maxAngles[h])
                if individuals[i,h]>maxAngles[h]:
                    individuals[i,h] =maxAngles[h] - dif
                elif individuals[i,h]<minAngles[h]:
                    individuals[i,h] =minAngles[h] + dif

