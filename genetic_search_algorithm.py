# 최적의 sequence
import numpy as np
#np.array([])

def sampling_tf(true_p, num):
    f_t=np.array([False, True])
    idx= np.random.choice(2, num, p= [1-true_p, true_p])
    idx_ft=f_t[idx]

    return idx_ft

# y_hat, input of model, output of model 의 데이터타입은 모두  numpy array (float)
def gen_search(model, y_hat, x_len, epoch=100, p_crossover= 0.2, p_mutation= 0.2, descendants=1000, max_p_clip=0.5, visible= True):
    #init
    best_gen= None
    best_loss= None

    gen= np.random.uniform(0, 1, [descendants,x_len])
    for ep in range(epoch):
        # loss 작을수록 sampling 확률 up
        y_pred= model(gen)



        loss= np.mean((y_pred - y_hat)**2, axis=0)

        sample_p= 1/ loss
        sample_p= sample_p/sample_p.sum()
        sample_p= np.clip(sample_p, 0, max_p_clip)
        sample_p= sample_p/sample_p.sum()



        idx= loss.argsort(axis=0)
        sample_p= sample_p[idx].reshape(-1)
        #print(sample_p)
        #print("")
        gen_sorted= gen[idx]
        loss_sorted= loss[idx]

        if ep==0:
            best_gen= gen_sorted[0]
            best_loss= loss_sorted[0]
        else:
            if best_loss >loss_sorted[0]:
                best_gen= gen_sorted[0]
                best_loss= loss_sorted[0]

        if (ep%(epoch//10)==0 or ep==epoch-1 ) and visible==True:
            #print("[", ep, "]", "loss:", loss_sorted[0], "gen", gen_sorted[0])
            print("[{0:5d}] best MSE loss:{1:15.20f} best gen:".format(ep, best_loss), best_gen)


        next_gen=None
        for i in range(gen_sorted.shape[0]//2):
            idx= np.random.choice(gen_sorted.shape[0], 2, p=sample_p, replace=False) # loss의 역수의 확률로 두개의 부모 뽑기
            gen_tmp= gen_sorted[idx]

            idx_ft= sampling_tf(p_crossover, x_len)

            #교차 실행
            for j in range(x_len):
                if idx_ft[j]==True:
                    tmp= gen_tmp[0, j]
                    gen_tmp[0, j]= gen_tmp[1, j]
                    gen_tmp[1, j]= tmp

            #두 자식을 next_gen 으로 전달
            if i==0:
                next_gen= gen_tmp
                #print("aa", next_gen.shape)
            else:
                next_gen= np.vstack([next_gen, gen_tmp])


        #돌연변이
        idx_ft= sampling_tf(p_crossover, [next_gen.shape[0],next_gen.shape[1]])
        random_gen= np.random.uniform(0, 1, [next_gen.shape[0],next_gen.shape[1]])

        next_gen= np.logical_not(idx_ft) * next_gen + idx_ft * random_gen




        # next_gen 이 현재의 gen이 됨
        gen= next_gen


    return loss_sorted[0], gen_sorted[0]
