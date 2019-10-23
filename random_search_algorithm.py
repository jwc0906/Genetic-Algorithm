
import numpy as np

def rand_search(model, y_hat, x_len, epoch=100, descendants=1000, visible= True):
    best_rand_x= None
    best_loss= None
    for ep in range(epoch):
        rand_x= np.random.uniform(0, 1, [descendants,x_len])
        y_pred= model(rand_x)
        loss= np.mean((y_pred - y_hat)**2, axis=0)

        idx= loss.argsort(axis=0)
        rand_x_sorted= rand_x[idx]
        loss_sorted= loss[idx]

        if ep==0:
            best_rand_x= rand_x_sorted[0]
            best_loss= loss_sorted[0]
        else:
            if best_loss >loss_sorted[0]:
                best_rand_x= rand_x_sorted[0]
                best_loss= loss_sorted[0]

        if (ep%(epoch//10)==0 or ep==epoch-1 ) and visible==True:
            #print("[", ep, "]", "loss:", loss_sorted[0], "gen", gen_sorted[0])
            print("[{0:5d}] best MSE loss:{1:15.20f} best gen:".format(ep, best_loss), best_rand_x)

    return best_loss, best_rand_x
