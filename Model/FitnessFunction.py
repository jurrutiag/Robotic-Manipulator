import RoboticManipulator
import numpy as np


#retorna pocisiones cartesionas de las tres masas móviles del brazo
def getPositions(population, manipulator):

    positions=[]

    for ind in population:

        ind_mat = ind.getGenes()

        # a2x, a2y, a2z, a3x, a3y, a3z, a4x, a4y, a4z

        position = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]])

        for ang in range(ind_mat.shape[0]):
            theta_1, theta_2, theta_3, theta_4 = ind_mat[ang,:]

            position_2, position_3, position_4 = manipulator.anglesToPositions(theta_1, theta_2, theta_3, theta_4)

            thisPosition= np.hstack([position_2, position_3, position_4])

            position = np.vstack([position, thisPosition])
        position= np.delete(position, 0, 0) #primera fila, eje 0
        positions.append(position)

    return positions

def getAccelerations(positions):

    accelerations = []

    for ind in positions:
        for i, pos in enumerate(ind):



#asume pesos iguales
def getCenterOfMass(positionOfWeigths):
    quantity=0
    COM=[0,0,0]
    for weight in positionOfWeigths:
        quantity+=1
        COM[0]+=positionOfWeigths[weight][0]
        COM[1]+=positionOfWeigths[weight][1]
        COM[2]+=positionOfWeigths[weight][2]
    COM[0]/=quantity
    COM[1]/=quantity
    COM[2]/=quantity
    return COM





                    