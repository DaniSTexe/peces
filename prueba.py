import numpy as np

Actual = np.array([(-1,-1,-1,-1,-1,-1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')])
Agregados = np.array([(-1,1,-1,-1,-1,-1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')])
Actual = np.append(Actual, Agregados, axis=0)
Agregados = np.array([(-1,2,-1,-1,-1,-1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')])
Actual = np.append(Actual, Agregados, axis=0)
Agregados = np.array([(-1,3,-1,-1,-1,-1)],  dtype=[('x', 'f8'),('y', 'f8'), ('w', 'f8'), ('h', 'f8'),('id', 'i4'),('valid', 'i4')])
Actual = np.append(Actual, Agregados, axis=0)

Actual = np.delete(Actual,0)
Actual = np.delete(Actual,0)
Actual = np.delete(Actual,0)
Actual = np.delete(Actual,0)
print(Actual)