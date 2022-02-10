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
y, N, Nframes =[(0.10),(0.12,0.08,0.05,0.04)],(1,4),2

#Main
y_line = 0.75

Anterior = np.array([(-1,-1,-1,-1,-1,-1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')])
Actual = np.array([(-1,-1,-1,-1,-1,-1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')])

identidad = 0
anteriorIdentificado = False
conteo_saltos_linea = 0
is_salto_linea = False
nuevos_ids = []
viejos_ids = []

def limpiadorOrdenador (Frame_arrays):
    Arrays_no_valids = ([-1])
    for i in range(len(Frame_arrays)):
        if Frame_arrays[i]['valid'] == -1:
            Arrays_no_valids.append(i)
            
    Frame_arrays_clean = np.delete(Frame_arrays,Arrays_no_valids[1:])
    menor_mayor = np.sort(Frame_arrays_clean, order='y')
    mayor_menor = menor_mayor[::-1]
    Arrays_no_valids = ([-1])
    return mayor_menor

Anterior = limpiadorOrdenador(Anterior)

for i0 in range(Nframes):
    
    if type(N) == tuple:
        N0 = N[i0]
    else:
        N0 = N

    for i1 in range(N0):
        if type(y) == tuple:
            y0 = y[i1]
        elif type(y) == list:
            try:
                y0 = y[i0][i1]
            except:
                y0 = y[i0]
        else:
            y0 = y

        out_Yolo = np.array([(-1,y0,-1,-1,-1,1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')]) 
        Actual = np.append(Actual, out_Yolo, axis=0)
        
        #Salida del frame ordenado 
        Actual = limpiadorOrdenador(Actual)
        i1=+1

    
    #Y_line posicion
    if y_line > Actual[-1]['y']:
        conteo_saltos_linea =+1 
        puntoA = Actual[-1]['y'] 
        puntoB = y_line
        is_salto_linea = True
    
    y_line = Actual[-1]['y']

    if Actual[0]['y'] > 0.75 and is_salto_linea == False:
        y_line = 0.75

    if is_salto_linea:
        #Identificador, encargado de colocar los IDs en cada frame:
        if Actual.size != 0:
            for i1 in range(len(Actual)):
                if Actual[i1]['id'] == -1:
                    if (Actual[i1]['y']<puntoB) and ((Actual[i1]['y'])>=puntoA):
                        identidad+=1
                        Actual[i1]['id'] = identidad
                        nuevos_ids.append(identidad)
                    else:
                        Actual[i1]['id'] = viejos_ids[i1]

    #Recordador de IDs, recuerda los peces que ya tienen ID Version 2.0
    else:
        for i2 in range(len(Actual)):
            Actual[i2]['id'] = identidad-len(Actual)+1+i2
    
    #Recordador de IDs, recuerda los peces que ya tienen ID Version 1.0
    """if anteriorIdentificado == True:
        for i2 in range(len(Actual)):
            Actual[i2]['id'] = identidad-len(Actual)+1+i2
    
    """

    print(f' Anterior: {Anterior}')
    print(f' Actual: {Actual}')

    #print(f'punto A {puntoA}')
    #print(f'punto B {puntoB}')
    #print(f'y line {y_line}')
    #print(f'salto linea {is_salto_linea}')

    #Reset
    viejos_ids = nuevos_ids
    nuevos_ids = []
    Anterior = Actual 
    Actual = np.array([(-1,-1,-1,-1,-1,-1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')]) 
    i0 =+ 1
    is_salto_linea = False


