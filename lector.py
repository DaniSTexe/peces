""" 
Codigo hecho para leer los textos guardados de un YOLO

Parametros de ejecución:
    Ingrese ruta de la carpeta donde estan los .txt
"""
import os
import numpy as np
from natsort import natsorted

path = input("Ingrese ruta de la carpeta donde estan los .txt ") #pide path de entrada al usuario
contenido = os.listdir(f'{path}') #leer contenido de la carpeta
contenido = natsorted(contenido) #libreria para que ordene los string como si fueran int
#print(contenido) #visualizar contenido 

for i in range(len(contenido)):
    f = open (f'{path}/{contenido[i]}','r')
    mensaje = f.read()
    #print(mensaje)
    #frameactual = np.array([mensaje[1:-2]])
    #frameactual = np.array([mensaje[1:-2]])

    print(f'-------------')
    print(f'--FRAME-{contenido[i]}--')
    frameactual = np.array([mensaje[2:-2]])
    print(frameactual)
    f.close()