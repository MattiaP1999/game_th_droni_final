import numpy as np
import scipy as sp
import numpy.matlib
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
def playGame(num_realizations,ch,strategy_alice,strategy_eve,x1,x2,tolerance,k,varphiprime): #[p_md,avg_distance] 
    realizations_Alice = np.int32(rand_sample(strategy_alice,num_realizations,tolerance))[0,:]
    strategy_eve[strategy_eve<tolerance]=0
    realizations_Eve = np.int32(rand_sample(strategy_eve,num_realizations,tolerance))[0,:]
    #realizations_Eve = realizations_Alice
    num_mis_det = 0
    real_eve = ch[0,realizations_Eve]
    att_eve=real_eve+np.random.normal(np.zeros((1,num_realizations)),real_eve/np.sqrt(k),None)
    dist =0
    prev_Alice = realizations_Alice[0]
    for ii in range(realizations_Eve.shape[0]):
        current_pos_Alice = realizations_Alice[ii]
        curr_channel_Alice = ch[0,current_pos_Alice]
        att_eve_fake = curr_channel_Alice+np.random.normal(np.zeros((1,1)),curr_channel_Alice/np.sqrt(k),None)
        d= (x1[0,current_pos_Alice]-x1[0,prev_Alice])**2+(x2[0,current_pos_Alice]-x2[0,prev_Alice])**2
        dist = dist+np.sqrt(d)
        prev_Alice = current_pos_Alice
        test = (k*(att_eve[0,ii]-curr_channel_Alice)**2)/(curr_channel_Alice**2)
        #test = (k*(att_eve_fake-curr_channel_Alice)**2)/(curr_channel_Alice**2)
        if test<varphiprime:
            num_mis_det = num_mis_det+1

    p_md = num_mis_det/realizations_Alice.shape[0]
    avg_distance = dist/realizations_Alice.shape[0]
    return p_md,avg_distance
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
def playGame_multiple(num_realizations,ch,strategy_alice,strategy_eve,x1,x2,tolerance,k,varphiprime,N): #[p_md,avg_distance]
    num_games =  num_realizations
    realizations_Alice = np.int32(rand_sample(strategy_alice,N*num_games,tolerance))[0,:]
    strategy_eve[strategy_eve<tolerance]=0
    realizations_Eve = np.int32(rand_sample(strategy_eve,N*num_games,tolerance))[0,:]
    #realizations_Eve = realizations_Alice
    num_mis_det = 0
    num_fa =0
    real_eve = ch[0,realizations_Eve]
    real_Alice = ch[0,realizations_Alice]
    att_eve=real_eve+np.random.normal(np.zeros((1,N*num_games)),real_eve/np.sqrt(k),None)
    att_Alice=real_Alice+np.random.normal(np.zeros((1,N*num_games)),real_Alice/np.sqrt(k),None)
    for ii in range(num_games):
        test_Eve=0
        test_Alice=0
        for rounds in range(N):
            current_pos_Alice = realizations_Alice[rounds+ii]
            curr_channel_Alice = ch[0,current_pos_Alice]
            test_Eve = test_Eve + (k*(att_eve[0,rounds+ii]-curr_channel_Alice)**2)/(curr_channel_Alice**2)
            test_Alice = test_Alice + (k*(att_Alice[0,rounds+ii]-curr_channel_Alice)**2)/(curr_channel_Alice**2)
            #test = (k*(att_eve_fake-curr_channel_Alice)**2)/(curr_channel_Alice**2)
        if test_Eve<varphiprime:
            num_mis_det = num_mis_det+1
        if test_Alice>varphiprime:
            num_fa = num_fa+1
    p_md = num_mis_det/num_games
    p_fa = num_fa/num_games
    return p_md,p_fa




# START OF SCRIPT
n_ch_realization = 20
#p_fas = [1e-2]
n_pfas = 20
p_fas = np.ndarray.tolist(np.linspace(1e-3,1e-1,n_pfas))
N = 3
sigmas = [10,13,16]
results_valgame = np.zeros((3,n_pfas))
num_realizations = np.int32(5e5)
tolerance = 1e-5
k=50
distance =50
ptax = 15
counter =0
index_sigma =0
for sigma in sigmas:
    index_pfa = 0
    for p_fa in p_fas:
        ch_real = np.arange(1,n_ch_realization+1,1, dtype=int)
        val_games = np.zeros((1,n_ch_realization), dtype=float)
        val_games_check = np.zeros((1,n_ch_realization), dtype=float)
        risparmio =np.zeros((1,n_ch_realization), dtype=float)
        for kk in ch_real:
            filename = './data_from_python/sigma/data_'+str(ptax)+'_'+str(distance)+'_'+str(sigma)+'_'+str(kk)+'.mat'
            mat_f_load = sp.io.loadmat(filename)
            #mat_f_load = sp.io.loadmat('./data_from_python/dist/data_gth_pythontemp.mat') 
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
            val_game,mix_row = game_solver(p_md)
            _,mix_col = game_solver(-p_md.transpose())
            mix_row[mix_row<tolerance]=0
            strategy_eve_naive = np.matlib.repmat(mix_row.transpose(),mix_row.shape[0],1) #repeted by rows
            strategy_Alice_naive = np.matlib.repmat(mix_col.transpose(),mix_col.shape[0],1) #repeted by rows
            p_md_estim_multiple,p_fa_estim_multiple=playGame_multiple(num_realizations,ch,strategy_Alice_naive,strategy_eve_naive,x1_line,x2_line,tolerance,k,varphi_prime_multiple,N)
            val_games_check[0][kk-1]=p_md_estim_multiple
        #results_valgame[index_sigma,index_dist]=val_games.mean()
        #results_risparmio[index_sigma,index_dist]=risparmio.mean()
        results_valgame[index_sigma,index_pfa]=val_games_check.mean()
        counter = counter +1
        print(str(counter/np.float16(results_valgame.size)*100))
        index_pfa=index_pfa+1
        filename ="Results_pfa_pmd_round_"+str(N)+".mat"
        dic = {"values_game_"+str(N):results_valgame,"p_fas":p_fas}
        sp.io.savemat(filename, dic)
    index_sigma=index_sigma+1
    # plotting
"""fig, ax = plt.subplots()


ax.semilogy(N, results_valgame[0,:], label="pfa=1e-4")
ax.semilogy(N, results_valgame[1,:], label="pfa=5e-4")
ax.semilogy(N, results_valgame[2,:], label="pfa=1e-3")

plt.xlabel('N') 
plt.ylabel('Pmd') 
plt.title('Pmd VS Number of rounds') 
plt.grid(True) 
plt.legend(loc="upper right")
plt.show()

print("Stop")"""



