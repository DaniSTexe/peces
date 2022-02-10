import numpy as np

Actual = np.array([(-1,0.1,-1,-1,-1,1),(-1,0.2,-1,-1,-1,1),(-1,0.3,-1,-1,-1,1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')])

# tenia 1,2,3
#agregue 4,5
identidad = []
Conteo_frame = 3

def limpiadorOrdenador (Frame_arrays):
        Arrays_no_valids = ([-1])
        for i in range(len(Frame_arrays)):
            if Frame_arrays[i]['valid'] == -1:
                Arrays_no_valids.append(i)
                
        Frame_arrays_clean = np.delete(Frame_arrays,Arrays_no_valids[1:])
        inferior_superior = np.sort(Frame_arrays_clean, order='y')
        #superor_inferior = inferior_superior[::-1]
        Arrays_no_valids = ([-1])
        return inferior_superior
    
Actual = limpiadorOrdenador(Actual)

print(f'ordenado {Actual}')



for i2 in range(1,Conteo_frame+1):
    identidad.append(i2)

Inverso_identidad = identidad[::-1]
print(Inverso_identidad)


for i3 in range(0, len(Inverso_identidad)):
    print(Actual[i3]['id'])
    Actual[i3]['id'] = Inverso_identidad[i3]
    print(Actual[i3]['id'])

print(Actual)


