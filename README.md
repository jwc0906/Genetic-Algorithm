# Random Search Algorithm

### example code
<code><pre>mse_loss, x= rand_search(model=model_example, y_hat=y_hat, x_len=4, epoch=10000, descendants=2000, visible= True)</code></pre>

# Genetic Search Algorithm

### example code
<code><pre>mse_loss, x= gen_search(model=model_example, y_hat=y_hat, x_len=4, epoch=100, p_crossover= 0.2, p_mutation= 0.2, descendants=1000, max_p_clip=0.5, visible= True)</code></pre>
