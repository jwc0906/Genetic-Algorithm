# Genetic Algorithm

### example code

<code><pre>
from genetic_algorithm import gen_algorithm
import numpy as np


def model_example(x):
    w_= np.array([[1, 5, 3, -1, -3]])
    result= np.sum(x * w_, axis=1)
    return result

y_hat= np.array([3])

mse_loss, x= gen_algorithm(model=model_example, y_hat=y_hat, x_len=5, epoch=100)

print(mse_loss)
print(x)</code></pre>
