import numpy as np

"""
Parametros
y: Posiciones en y de las detecciones en yolo. 
N: Numero de detecciones por frame.
Nframes: Numero de frames de la simulacion. (Sin contar el anterior), Es decir Nframes = 1 tendra el anterior y el nuevo, Nframes = 2 tendra el anterior y dos nuevos.

Ejemplos:
Una detección en un frame y,N,Nframes = 0.1,1,1 

Dos detecciones en un frame y,N,Nframes = (0.1,0.2),2,1 

Una detección en dos frames y,N,Nframes = [(0.1),(0.2)],(1,1),2

Tres detecciones por frame en 2 frames y,N,Nframes = [(0.1,0.1,0.1),(0.2,0.2,0.2)], (3,3), 2
"""

#Situaciones posibles:

#situacion 0: Entra un solo pez sin existir alguno en el anterior (Se debe colocar id 1)
#y,N, Nframes = 0.1,1,1
#Ok

#situacion 1: Entran 3 peces sin existir alguno en el anterior (Se debe colocar id 1,2,3 ascendentes en y)
#y, N, Nframes =(0.10,0.12,0.11),3,1
#Ok

#situacion 2: siguen los 3 peces su trayectoria normalmente (Se deben mantener los ids 1, 2, 3 ascendetes en y)
#y, N, Nframes =[(0.10,0.12,0.11),(0.76,0.77,0.78)],(3,3),2
#Ok funciona con N peces

#situacion 3: Pasada la situacion 0 entran N peces (N=4)
#y, N, Nframes =[(0.10),(0.12,0.08,0.05,0.04)],(1,4),2

#Situacion Especial
x=[(-1),(-1,-1),(-1,-1),(-1,-1,-1),(-1,-1,-1)]
y=[(0.2),(0.3,0.1),(0.7,0.5),(0.9,0.7,0.1),(0.91,0.2,0.1)]
N, Nframes =[(1,2,2,3,3),5]

#Definicion de variables
Anterior = np.array([(-1,-1,-1,-1,-1,-1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')])
Actual = np.array([(-1,-1,-1,-1,-1,-1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')])
pecesTotales = 0
pecesFrame = 0 
pezNuevo = 0
indexId = []
posicionDistancia = np.array([(-1,-1,-1)],  dtype=[('posicion', 'i4'),('distancia', 'f8'),('valid','i4')])

def limpiadorOrdenador (Frame_arrays, ordenar):
    Arrays_no_valids = ([-1])
    for i in range(len(Frame_arrays)):
        if Frame_arrays[i]['valid'] == -1:
            Arrays_no_valids.append(i)
            
    Frame_arrays_clean = np.delete(Frame_arrays,Arrays_no_valids[1:])
    menor_mayor = np.sort(Frame_arrays_clean, order=ordenar)
    #mayor_menor = menor_mayor[::-1]
    Arrays_no_valids = ([-1])
    return menor_mayor

Anterior = limpiadorOrdenador(Anterior,'y')

#Simulador
for i0 in range(Nframes):
    
    if type(N) == tuple:
        N0 = N[i0]
    else:
        N0 = N

    for i1 in range(N0):
        if type(y) == tuple:
            y0 = y[i1]
            x0 = x[i1]
        elif type(y) == list:
            try:
                y0 = y[i0][i1]
                x0 = x[i0][i1]
            except:
                y0 = y[i0]
                x0 = x[i0]
        else:
            y0 = y
            x0 = x

        out_Yolo = np.array([(x0,y0,-1,-1,-1,1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')]) 
        valid = 1
        Actual = np.append(Actual, out_Yolo, axis=0)
        
        #Salida del frame ordenado 
        Actual = limpiadorOrdenador(Actual,'y')

    #Frames anteriores y actuales del momento
    print(f'Anterior: {Anterior}')
    print(f'Actual: {Actual}')

    #Condición 1
    if len(Anterior) == 0:
        pecesFrame = len(Actual)

    #ActualVacio para poder elminar peces del vector actual sin afectar la logica
    ActualVacio = Actual 

    #Condicion 2
    for i3 in range(len(Anterior)):
        if i3 == 0:
            for i4 in range(len(Actual)): 
                if Anterior[i3]['y'] >= Actual[i4]['y']:
                    ActualVacio = np.delete(ActualVacio,i4)
                    pezNuevo += 1
            else:
                break
        else:
            try:
                ActualVacio = np.delete(ActualVacio,i3)
                #posicionDistancia = np.append(posicionDistancia, [(i3,distancia,1)], axis=0)
            except:
                print('Error 101')

    #Conteo de Frame
    pecesTotales = pecesTotales + pezNuevo 
    pecesTotales = pecesTotales + pecesFrame
    print(f'Conteo: {pecesTotales}')
    print(f'----------------------')

    #Reset
    Anterior = Actual 
    Actual = np.array([(-1,-1,-1,-1,-1,-1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')]) 
    pecesFrame = 0
    pezNuevo = 0
    