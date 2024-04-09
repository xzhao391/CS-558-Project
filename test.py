import numpy as np
folder = '../milestone2/path/path'
data = np.loadtxt(folder+str(2000)+'.dat', unpack=True)
print(data)