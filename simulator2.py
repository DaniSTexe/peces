import numpy as np

"""
Parametros
x: Posiciones en y de las detecciones en yolo. 
y: Posiciones en y de las detecciones en yolo. 
N: Numero de detecciones por frame.
Nframes: Numero de frames de la simulacion. 

Ejemplo:
#Tendremos 5 frames,los peces en cada frame seran (1,2,2,3,3) respectivamente, el pez del frame 1 estara en la posicion (0.15,0.2) y asi...
x=[(0.15),(0.15,0.23),(0.15,0.23),(0.15,0.23,0.81),(0.23,0.81,0.52)]
y=[(0.2),(0.3,0.1),(0.7,0.5),(0.9,0.7,0.1),(0.91,0.2,0.1)]
N, Nframes =[(1,2,2,3,3),5]
"""

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
pezDeMas = 0
Identificador = 0

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
                ActualVacio = np.delete(ActualVacio,0)
                print(f'Vacio: {len(ActualVacio)}')
                print("Borrado normal")
            except:
                print('Error 101')
                pezDeMas += 1

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
    pezDeMas = 0
    