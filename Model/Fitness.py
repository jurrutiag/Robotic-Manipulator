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




                    