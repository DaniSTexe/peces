import numpy as np

conteo2=0
idEncontrado=False

Anterior = np.array([(-1,0.2,-1,-1,4,-1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')])
Actual = np.array([(-1,0.7,-1,-1,4,-1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')])
Agregados = np.array([(-1,1,-1,-1,3,-1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')])
Actual = np.append(Actual, Agregados, axis=0)
Agregados = np.array([(-1,2,-1,-1,2,-1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')])
Actual = np.append(Actual, Agregados, axis=0)
Agregados = np.array([(-1,3,-1,-1,1,-1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')])
Actual = np.append(Actual, Agregados, axis=0)

posicion = np.where(Actual['id']== 1)

for i8 in range(6):
    print(f'id {i8}')
    try:
        #Encontramos el Id en el frame anterior y el actual
        idFrameAnterior = np.where(Anterior['id']== i8)
        idFrameActual = np.where(Actual['id']== i8)
        idEncontrado = True
    except:
        print("ID no encontrado") 
         
    if idEncontrado:
        idFrameAnterior = (Anterior[idFrameAnterior]['y'])
        idFrameActual = (Actual[idFrameActual]['y'])     

        if (idFrameAnterior.size > 0) and (idFrameActual.size > 0):
            if (idFrameAnterior < 0.5) and (idFrameActual > 0.5):
                conteo2 += 1
                

Actual = np.delete(Actual,0)
Actual = np.delete(Actual,0)
Actual = np.delete(Actual,0)
Actual = np.delete(Actual,0)
""" print(Actual) """

