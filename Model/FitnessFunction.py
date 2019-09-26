import RoboticManipulator


#retorna pocisiones cartesionas de las tres masas móviles del brazo
def getPositions(population, manipulator):

    positions=[]

    for ind in population:
        position = np.array([0,0,0])
        for ang in ind.shape(0):
            theta_1, theta_2, theta_3, theta_4 = ind[ang,:]

            pos_2, pos_3, pos_4 = manipulator.anglesToPosition(theta_1, theta_2, theta_3, theta_4)

            thisPosition= [position_2, position_3, position_4]

            np.vstack([position, thisPosition])
        position= np.delete(position, 0, 0) #primera fila, eje 0
        positions.append(position)

    return positions

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





                    