import numpy as np

def IsInCollision(x,obc):
    if (x[0] > 20 or x[0] < -20) or (x[1] > 20 or x[1] < -20) \
            or (x[2] > 20 or x[2] < -20):
        return True
    size_list = [(5,5,10),(5,10,5),(5,10,10),(10,5,5),(10,5,10),(10,10,5),
                 (10, 10, 10), (5, 5, 5), (10, 10, 10), (5,5,5)]
    s=np.zeros(3,dtype=np.float32)
    s[0]=x[0]
    s[1]=x[1]
    s[2]=x[2]
    for i in range(0,10):
        collision = True
        for j in range(0,3):
            if abs(obc[i][j] - s[j]) > size_list[i][j]/2.0:
                collision = False
                break
        if collision == True:
            return True
    return False