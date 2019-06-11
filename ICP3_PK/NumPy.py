import numpy as np
x = np.random.randint(1, 20, 15)
print("List of integers generated randomly: ", x)
x[x.argmax()] = 0
print("Maximum value within the vector is replaced by 0. The new vector list is ", x)
