from genetic_search_algorithm import gen_search
from random_search_algorithm import rand_search
import numpy as np


def model_example(x):
    w_= np.array([[0.1, 0.5, 0.3, -0.1], [0.1, 0.1, 0.1, -0.1], [0.6, -0.5, 0.1, -0.1]])
    result= np.matmul(w_, x.T)
    return result

y_hat= np.array([[0.3], [0.4], [0.9]])


print("gen_search")
mse_loss, x= gen_search(model=model_example, y_hat=y_hat, x_len=4, epoch=100, p_crossover= 0.2, p_mutation= 0.2, descendants=1000, max_p_clip=0.5, visible= True)
print(mse_loss)
print(x)

print("rand_search")
mse_loss, x= rand_search(model=model_example, y_hat=y_hat, x_len=4, epoch=10000, descendants=2000, visible= True)
print(mse_loss)
print(x)
