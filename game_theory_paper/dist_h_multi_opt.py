import numpy as np
import scipy as sp
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def get_distance(x1,x2):
    len = int(x1.size)
    dist =np.zeros((1,(len*len)))
    index_pos = 0
    for ii in range(len):
        for jj in range(len):
            d= (x1[0][ii]-x1[0][jj])**2+(x2[0][ii]-x2[0][jj])**2
            dist[0][index_pos]=np.sqrt(d)
            index_pos=index_pos+1
    return dist

def greedyStrategy(A_ub,b_ub,A_eq,b_eq,c,lb,ub):
    bounds=np.array([lb[:,0], ub[:,0]]).transpose()
    res=sp.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', callback=None, options=None, x0=None, integrality=None)
    local_ne = res['x']
    if(res['status']!=0):
        print("Error in greedy")
        local_ne = -1
    return [local_ne,res['status']]

def game_solver(A): 
    r=A.shape[0]
    c = A.shape[1]
    AA = np.ones((r,c+1))
    AA[0:r,0:c] = -A.transpose()
    Aeq = np.ones((1,r+1))
    Aeq[0,-1]=0
    b = np.zeros((c,1))
    beq = 1
    lb = np.zeros((r+1,1))
    lb[-1]=-1
    ub = np.ones((r+1,1))
    f = np.zeros((r+1,1))
    f[-1]=-1
    [local_ne,flag] = greedyStrategy(AA,b,Aeq,beq,f,lb,ub)
    if flag==0:
        p = local_ne[0:r]
        v = local_ne[r]
    else:
        p = -1
        v = -1
    return v,p

def rand_sample(probabilities,num_sim, tolerance):
    probabilities[probabilities<tolerance]=0
    current_position = np.random.randint(0,probabilities.shape[0])
    arr = np.ndarray.tolist(np.arange(0,probabilities.shape[0],1))
    y= np.ones((1,num_sim))
    for ii in range(num_sim):
        y[0][ii]=current_position
        p = probabilities[current_position,:]
        p=p/np.sum(p)
        p = p.transpose()
        current_position = np.random.choice(arr,None,True,p)
    return y 
def playGame(num_realizations,ch,strategy_alice,strategy_eve,x1,x2,tolerance,k,varphiprime,strategy_alice_naive): #[p_md,avg_distance] 
    realizations_Alice = np.int32(rand_sample(strategy_alice,num_realizations,tolerance))[0,:]
    realizations_Alice_naive = np.int32(rand_sample(strategy_alice_naive,num_realizations,tolerance))[0,:]
    strategy_eve[strategy_eve<tolerance]=0
    realizations_Eve = np.int32(rand_sample(strategy_eve,num_realizations,tolerance))[0,:]
    num_mis_det = 0
    real_eve = ch[0,realizations_Eve]
    att_eve=real_eve+np.random.normal(np.zeros((1,num_realizations)),real_eve/np.sqrt(k),None)
    dist =0
    dist_naive =0
    prev_Alice = realizations_Alice[0]
    prev_Alice_naive = realizations_Alice_naive[0]
    gain_temp =0
    for ii in range(realizations_Eve.shape[0]):
        current_pos_Alice = realizations_Alice[ii]
        current_pos_Alice_naive = realizations_Alice_naive[ii]
        curr_channel_Alice = ch[0,current_pos_Alice]
        #att_eve_fake = curr_channel_Alice+np.random.normal(np.zeros((1,1)),curr_channel_Alice/np.sqrt(k),None)
        d= (np.float64(x1[0,current_pos_Alice])-np.float64(x1[0,prev_Alice]))**2+(np.float64(x2[0,current_pos_Alice])-np.float64(x2[0,prev_Alice]))**2
        d_naive = (np.float64(x1[0,current_pos_Alice_naive])-np.float64(x1[0,prev_Alice_naive]))**2+(np.float64(x2[0,current_pos_Alice_naive])-np.float64(x2[0,prev_Alice_naive]))**2
        dist = dist+np.sqrt(d)
        dist_naive = dist_naive+np.sqrt(d_naive)
        prev_Alice = current_pos_Alice
        prev_Alice_naive = current_pos_Alice_naive
        test = (k*(att_eve[0,ii]-curr_channel_Alice)**2)/(curr_channel_Alice**2)
        gain_temp = gain_temp+(np.sqrt(d_naive)-np.sqrt(d))
        #test = (k*(att_eve_fake-curr_channel_Alice)**2)/(curr_channel_Alice**2)
        if test<varphiprime:
            num_mis_det = num_mis_det+1

    p_md = num_mis_det/realizations_Alice.shape[0]
    avg_distance = dist/(realizations_Alice.shape[0]-1)
    avg_dist_naive = dist_naive/(realizations_Alice.shape[0]-1)
    gain = gain_temp/(realizations_Alice.shape[0]-1)
    return p_md,avg_distance,avg_dist_naive,gain

def find_opt_nes(A,D,value_g,tolerance):
    r=A.shape[0]
    c = A.shape[1]
    b = -(-value_g-tolerance)*np.ones((c,1))
    opt_nes = np.zeros((r,r))
    for ii in range(r):
        f= D[ii,:].transpose()
        Aeq = np.ones((1,r))
        lb = np.zeros((r,1))
        ub = np.ones((r,1))
        beq =1
        [local_ne,flag] = greedyStrategy(A,b,Aeq,beq,f,lb,ub)
        opt_nes[:,ii]=local_ne
    return opt_nes
def getVariance(mean,vector,N):
    var = 0
    for el in vector[0,:]:
        var = var + (el-mean)**2
    return var/N



# START OF SCRIPT

initial_position = 150
print("Initial pos:")
print(initial_position)
n_ch_realization = 50
p_fa = 1e-3
distances = [10,20,30,50,70,100]
n_dist= len(distances)
results_valgame = np.zeros((1,n_dist))
results_dist_naive = np.zeros((1,n_dist))
results_dist_opt = np.zeros((1,n_dist))
results_variance_naive = np.zeros((1,n_dist)) #variance
results_variance_opt= np.zeros((1,n_dist)) #variance
num_realizations = np.int32(2e5)
tolerance = 1e-4
k=50
N=1
ptax = 15
sigmas =[10]
counter =0
index_sigma =0


# Variance
filename_variance_one = './results/data/distance_altezza/average_single.mat'
filename_variance_multi = './results/data/distance_altezza/average_multi.mat'
load_var = sp.io.loadmat(filename_variance_one)
mean_opt_one = load_var['average_res_opt_single'][0,:]
mean_naive_one = load_var['average_res_naive_single'][0,:]

load_var = sp.io.loadmat(filename_variance_multi)
mean_opt_multi = load_var['average_res_opt_multi'][0,:]
mean_naive_multi = load_var['average_res_naive'][0,:]
mean_naive= (mean_naive_one+mean_naive_multi)/2

for sigma in sigmas:
    distance_index = 0
    for dist in distances:
        ch_real = np.arange(0,n_ch_realization,1, dtype=int)
        val_games = np.zeros((1,n_ch_realization), dtype=float)
        risparmio =np.zeros((1,n_ch_realization), dtype=float)
        avg_dist_naive = np.zeros((1,n_ch_realization), dtype=float)
        avg_dist_opt = np.zeros((1,n_ch_realization), dtype=float)
        temp_variance_naive = np.zeros((1,n_ch_realization), dtype=float)
        temp_variance_opt = np.zeros((1,n_ch_realization), dtype=float)
        for kk in ch_real:
            filename = './data_from_python/complete_dataset/data_'+str(ptax)+'_'+str(dist)+'_'+str(sigma)+'_'+str(kk+initial_position)+'.mat'
            mat_f_load = sp.io.loadmat(filename)
            ch = (10**(-np.array(mat_f_load["ch"])/20))**2
            ch = ch+min(min(ch)/10)
            x1 = np.array(mat_f_load["x1"])
            x2 = np.array(mat_f_load["x2"])
            n_att = ch.shape[1]
            x1_line = np.reshape(x1,(1,-1))
            x2_line = np.reshape(x2,(1,-1))
            dist_matrix = np.reshape(get_distance(x1_line,x2_line),[x1_line.size,x1_line.size])
            varphi_prime= sp.stats.chi2.ppf(1-p_fa, 1, loc=0, scale=1)
            varphi_prime_multiple= sp.stats.chi2.ppf(1-p_fa, N, loc=0, scale=1)
            #gi_star= np.zeros((n_att,n_att))
            p_md = np.zeros((n_att,n_att)) 
            for ii in range(n_att):
                for jj in range(n_att):
                    nc = ((ch[0][ii]-ch[0][jj])*np.sqrt(k)/ch[0][ii])**2
                    x=varphi_prime*(ch[0][jj]/ch[0][ii])**2
                    p_md[ii,jj] = sp.stats.ncx2.cdf(x, 1, nc, loc=0, scale=1)
            p_md[p_md<tolerance]=0
            val_game,mix_row = game_solver(p_md) #Single round
            _,mix_col = game_solver(-p_md.transpose()) #Single round
            mix_row[mix_row<tolerance]=0
            mix_col[mix_col<tolerance]=0
            
            # Distance           
            strategy_eve_naive = np.matlib.repmat(mix_row.transpose(),mix_row.shape[0],1) #repeted by rows
            strategy_Alice_naive = np.matlib.repmat(mix_col.transpose(),mix_col.shape[0],1) #repeted by rows
            #Multiple games
            val_games[0,kk-1]=val_game
            #Solve strategy
            strategy_Alice = find_opt_nes(p_md,dist_matrix,val_game,tolerance).transpose()
            [p_md_estim,avg_distance_estim,avg_distance_naive,gain] = playGame(num_realizations,ch,strategy_Alice,strategy_eve_naive,x1_line,x2_line,tolerance,k,varphi_prime,strategy_Alice_naive)
            risparmio[0][kk-1] = gain
            avg_dist_naive[0][kk-1] = avg_distance_naive
            avg_dist_opt[0][kk-1] = avg_distance_estim
        results_valgame[index_sigma,distance_index]=val_games.mean()
        results_dist_naive[index_sigma,distance_index]=avg_dist_naive.mean()
        results_dist_opt[index_sigma,distance_index]=avg_dist_opt.mean()

        #Variance
        results_variance_naive[index_sigma,distance_index]=getVariance(mean_naive[distance_index],avg_dist_naive,n_ch_realization)
        results_variance_opt[index_sigma,distance_index]=getVariance(mean_opt_multi[distance_index],avg_dist_opt,n_ch_realization)

        filename = "results_multi_opt_dist_variance"+str(initial_position)+".mat"
        dictionary = {"values_game_"+str(initial_position):results_valgame, "results_dist_opt_multi"+str(initial_position):results_dist_opt, "results_dist_naive_multi"+str(initial_position):results_dist_naive,"results_variance_naive_multi"+str(initial_position):results_variance_naive,"results_variance_opt_multi"+str(initial_position):results_variance_opt}
        sp.io.savemat(filename,dictionary) 
        distance_index=distance_index+1
        print(distance_index)
    index_sigma=index_sigma+1