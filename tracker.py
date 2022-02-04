import math
import numpy as np

#Situacion 0: Frame anterior no tiene nada y el actual acaba de detectar 
#Situacion 1: frame anterior es el primer frame de la imagen en detectar y tenemos igual numero de detecciones en el anterior como en el actual 
#Situacion 2: Frame anterior tiene menos detecciones que en el frame actual ( Es decir aparecen nuevos peces en la pantalla)
#Situacion 3: Frame anterior tiene mas detecciones que el frame actual (Es decir peces salieron de la pantalla)
#Situacion 4: El detector falla y deja de detectar uno o varios peces  que realmente si estan aun en la pantallas 
              #(Solucion: Agregar un tercer o cuarto o hasta 10 vectores para hacer validacion de que fue solo un error del detector, minimo con 3)
#Situacion 5: Todas las situaciones pero variando el numero de vectores de entrada

#Situacion 0: OK
"""
try:    
    frameanterior = ([(0, 0, 0, 0)])
except:
    frameanterior = ([(0, 0, 0, 0)])
frameactual = np.array([(0.5, 0.62, 0.1, 0.1),(0.5, 0.76,0.1, 0.1)])"""

"""
#Situacion 1: OK
frameanterior = np.array([(0.5, 0.1, 0.1, 0.1),(0.5, 0.5,0.1, 0.1)])
frameactual = np.array([(0.5, 0.2, 0.1, 0.1),(0.5, 0.7,0.1, 0.1)])"""

#Situacion 2: Parece OK
"""frameanterior = np.array([(0.5, 0.6, 0.1, 0.1),(0.5, 0.75,0.1, 0.1)])
frameactual = np.array([(0.5, 0.62, 0.1, 0.1),(0.5, 0.76,0.1, 0.1),(0.5, 0.1,0.1, 0.1)])"""

#Situacion 3: OK
frameanterior = np.array([(0.5, 0.1, 0.1, 0.1),(0.5, 0.5,0.1, 0.1),(0.5, 0.7,0.1, 0.1)])
frameactual = np.array([(0.5, 0.31, 0.1, 0.1),(0.5, 0.85,0.1, 0.1)])

"""frameanterior = np.array([(0.5, 0.1, 0.1, 0.1)])
frameactual = np.array([(0.5, 0.2, 0.1, 0.1),(0.5, 0.7,0.1, 0.1)])"""

#Main
ternas_frame = np.array([(-1, -1, -1)], dtype=[('id', 'i4'),('distancia', 'f8'),('posicion','i4')])
xywh_array = np.array([(-1, -1, -1, -1, -1)], dtype=[('x', 'f8'),('y', 'f8'),('w', 'f8'),('h', 'f8'),('id', 'f8')])

def identador(frame):
    n = len(frame)
    creador = np.arange(1,(n+1)).reshape((n,1))
    frame = np.append(frame, creador, axis=1)
    return frame

def distancias(x1,x2,y1,y2):
    """print(f' x1 es {x1}')
    print(f' x2 es {x2}')
    print(f' y1 es {y1}')
    print(f' y2 es {y2}')"""
    distancia = math.sqrt(((x2-x1)*(x2-x1))+((y2-y1)*(y2-y1)))
    return distancia

def menorDistancia(frame_completo):
    frame_completo = np.sort(frame_completo, order='distancia')
    salida = ( frame_completo[1][0], frame_completo[1][1], frame_completo[1][2])
    return salida

def envioTuplas(terna):
    global ternas_frame
    nuevo = np.array([terna],  dtype=[('id', 'i4'),('distancia', 'f8'),('posicion','i4')])
    ternas_frame = np.append(ternas_frame, nuevo, axis=0)

def frameFinalizado(indicador):
    global ternas_frame
    if indicador == len(frameactual):
        #print("Frame Terminado")
        menor = menorDistancia(ternas_frame)
        ternas_frame = np.array([(-1, -1, -1)], dtype=[('id', 'i4'),('distancia', 'f8'),('posicion','i4')])
        return menor

#print(f'Shape: {frameanterior.shape}')
frameanterior = identador(frameanterior)
frameactual = identador(frameactual)


#Situacion donde desaparece un pez (Se elimina la posición en Y mas proxima)
if len(frameanterior) > len(frameactual):
    diferencia = len(frameanterior)-len(frameactual)
    for i in range(len(frameanterior)):
        nuevo = (frameanterior[i][0],frameanterior[i][1], frameanterior[i][2], frameanterior[i][3], frameanterior[i][4])
        Convertido = np.array([nuevo], dtype=[('x', 'f8'),('y', 'f8'),('w', 'f8'),('h', 'f8'),('id', 'f8')])
        xywh_array = np.append(xywh_array, Convertido, axis=0)

    Ordenado = np.sort(xywh_array, order='y')
    m = len (frameanterior)
    frameanterior = (Ordenado[1:m+1-diferencia])

for i in range(len(frameanterior)):
    #print(f'Frame anterior posicion: {i}')

    for j in range(len(frameactual)):
        if frameanterior[i][1] < frameactual[j][1]:
            resultado = distancias(frameanterior[i][0], frameactual[j][0], frameanterior[i][1], frameactual[j][1])
            #print(f'Frame acutal posicion: {j}, distancia: {resultado}')
            envioTuplas((frameanterior[i][4],resultado, j))
            #print(ternas_frame)
            menor = frameFinalizado(j+1)
            if menor != None:
                frameactual[(menor[2])][4] = menor[0]
    
    #asignador de frames al frameactual

print(f'frameanterior {frameanterior}')
print(f'frameactual {frameactual}')
