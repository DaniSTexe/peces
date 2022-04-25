from re import A
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
ultimoId = 0
ultimoIdAsignador = 0
conteo2 = 0
idEncontrado=False

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

    #ActualVacio para poder elminar peces del vector actual sin afectar la logica
    ActualVacio = Actual 
    #Esto se hace para mantener el valor del ultimo Id
    ultimoIdAsignador = ultimoId

    #Condición 1
    if len(Anterior) == 0:
        pecesFrame = len(Actual)
        for i5 in range(len(Actual)):
            ultimoId +=1
            ultimoIdAsignador +=1

    #Necesitamos conocer los ids de los peces del anterior frame para partir desde el ultimo id
    for i6 in range(len(Anterior)):
        #Recibimos el id de cada pez anterior y lo comparamos obteniendo el mayor valor
        if ultimoId < Anterior[i6]['id']:
            ultimoId = Anterior[i6]['id']

    print(f'ultimoID {ultimoId}')

    #Condicion 2
    for i3 in range(len(Anterior)):
        if i3 == 0:
            for i4 in range(len(Actual)): 
                if Anterior[i3]['y'] >= Actual[i4]['y']:
                    ActualVacio = np.delete(ActualVacio,i4)
                    pezNuevo += 1
                    ultimoId +=1
                    ultimoIdAsignador +=1
                    #Actual[i4]['id'] = ultimoId No lo asignamos aca porque si hay varios peces nuevos le asignara en orden inverso el id
                else:
                    #En este punto para la posicion 0 del vector anterior no hay ningun pez antes
                    #Se sigue con el procedimiento normal y se agrega aca para que se aplique al primer pez de anterior
                    #es decir desde las siguientes corchetes se va a repetir en el else de abajo (para los demas peces)

                    #{
                    try:
                        ActualVacio = np.delete(ActualVacio,0)
                        print(f'Vacio: {len(ActualVacio)}')
                        print("Borrado normal")
                        break
                    except:
                        #un pez aca significa que ya salio
                        print('Pez sale')
                        pezDeMas += 1
                        break
                    #}
        else:
            try:
                ActualVacio = np.delete(ActualVacio,0)
                print(f'Vacio: {len(ActualVacio)}')
                print("Borrado normal")
            except:
                #un pez aca significa que ya salio
                print('Pez sale')
                pezDeMas += 1

    #Esto con el fin de que si ya entro a la condición 1, los ids ya vienen correctamente, no sería necesario hacer esta parte del codigo
    
    for i7 in range(len(Actual)):
        #Enlazar ids
        Actual[i7]['id'] = ultimoIdAsignador
        ultimoIdAsignador -= 1

        #Esto significa que sobraron Ids 
        #Posibles escenarios
        #1) Hubo un pez nuevo que no fue contado con la condicion de menor altura
        #2) Peces del frame anterior ya salieron
        if ultimoIdAsignador <= -1:
            print(f"Problema con ID")
                    
    #Forma 2 de contar
    for i8 in range(ultimoId):
        try:
            #Encontramos el Id en el frame anterior y el actual
            idFrameAnterior = np.where(Anterior['id']== i8+1)
            idFrameActual = np.where(Actual['id']== i8+1)
            idEncontrado = True
        except:
            print("ID no encontrado") 
         
        if idEncontrado:
            idFrameAnterior = (Anterior[idFrameAnterior]['y'])
            idFrameActual = (Actual[idFrameActual]['y'])     

            if (idFrameAnterior.size > 0) and (idFrameActual.size > 0):
                if (idFrameAnterior < 0.5) and (idFrameActual > 0.5):
                    conteo2 += 1

    #Frames anteriores y actuales del momento
    print(f'Anterior: {Anterior}')
    print(f'Actual: {Actual}')

    #Conteo de Frame
    pecesTotales = pecesTotales + pezNuevo 
    pecesTotales = pecesTotales + pecesFrame
    print(f'Conteo: {pecesTotales}')
    print(f'----------------------')
    print(f'El conteo 2 es: {conteo2}')
    print(f'----------------------')

    #Reset
    Anterior = Actual 
    Actual = np.array([(-1,-1,-1,-1,-1,-1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')]) 
    pecesFrame = 0
    pezNuevo = 0
    pezDeMas = 0
    idEncontrado=False
    

#Estado actual
#Cuando llega a 0 los peces, se debe reiniciar el ultimoId y ademas guardar ese ultimoId que teniamos en una variable para corroborar, teniendo en cuenta que cada vez que entre a la condición 1, esos peces se deben agregar a la cuenta.
#no esta funcionando pero porque esta simulacion no es realista al asumir los saltos tan grandes de los peces, se debe o ajustar la simulación o 
    #ajustar los parametros para simular, o probrar directamente en el detect