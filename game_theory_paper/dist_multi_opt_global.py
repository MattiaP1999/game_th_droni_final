import numpy as np
import scipy as sp
import numpy.matlib
import os
import matplotlib.pyplot as plt

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

def game_solver_one_opt(A,D,value_g,tolerance):
    r=A.shape[0]
    c = A.shape[1]
    Aeq = np.ones((1,r))
    b = -(-value_g-tolerance)*np.ones((c,1))
    beq = 1
    lb = np.zeros((r,1))
    ub = np.ones((r,1))
    return oneOpt(A,b[0,:],Aeq,beq,D,lb[0,:],ub[0,:])
    #return oneOptGenetic(A,b[0,:],Aeq,beq,D,lb[0,:],ub[0,:])
def objective(x, D):
    temp=np.matmul(x.transpose(),D)
    return np.matmul(temp,x)
def quadhess(x,D,a):
    return 2*D
def grad(x,D):
    return 2*np.matmul(D,x)
def oneOpt(A_ub,b_ub,A_eq,b_eq,D,lb,ub,len_vec):
    bounds=np.array([lb, ub]).transpose()
    eq_constraint=sp.optimize.LinearConstraint(A=A_eq,lb=b_eq,ub=b_eq,keep_feasible=False)
    neq_constraint = sp.optimize.LinearConstraint(A_ub,-np.inf*np.ones(np.array(ub).shape),b_ub)

    termination_f = -1
    n_att = 1
    num_multi_start = 20
    best_obj_val = np.inf
    at_least_one = False
    while n_att<=num_multi_start or at_least_one==False:
        x0=np.random.rand(len_vec,)
        x0=x0/np.sum(x0)
        res=sp.optimize.minimize(objective, x0=x0, args=(D), method=None, jac=grad, hessp=None, bounds=bounds, constraints=(eq_constraint,neq_constraint), tol=1e-5, callback=None, options=None)
        termination_f = res['status']
        if termination_f==0:
            at_least_one = True
            obj_val = res['fun']
            if obj_val<best_obj_val:
                local_ne = res['x']
                best_obj_val = obj_val
        n_att = n_att+1
    return local_ne

def rand_sample_boosted(probabilities_one,probabilities_two,num_sim, tolerance):
    probabilities_one[probabilities_one<tolerance]=0
    probabilities_two[probabilities_two<tolerance]=0
    current_position = np.random.randint(0,probabilities_one.shape[0])
    arr = np.ndarray.tolist(np.arange(0,probabilities_one.shape[0],1))
    y= np.ones((num_sim,2))
    for ii in range(num_sim):
        p = probabilities_one[current_position,:]
        p=p/np.sum(p)
        p = p.transpose()
        current_position = np.random.choice(arr,None,True,p)
        y[ii][0]=current_position
        p_2 = probabilities_two[current_position,:]
        p_2=p_2/np.sum(p_2)
        p_2 = p_2.transpose()
        current_position = np.random.choice(arr,None,True,p_2)
        y[ii][1] = current_position
    return y
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
    #realizations_Eve = realizations_Alice
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
        att_eve_fake = curr_channel_Alice+np.random.normal(np.zeros((1,1)),curr_channel_Alice/np.sqrt(k),None)
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

def playGame_multiple_dist(num_realizations,ch,strategy_alice_one,strategy_alice_two,strategy_eve,x1,x2,tolerance,k,varphiprime): #[p_md,avg_distance]
    num_games =  num_realizations
    realizations_Alice = np.int32(rand_sample_boosted(strategy_alice_one,strategy_alice_two,num_games,tolerance))
    strategy_eve[strategy_eve<tolerance]=0
    realizations_Eve= np.int32(rand_sample_boosted(strategy_eve,strategy_eve,num_games,tolerance))
    #realizations_Alice = np.zeros(np.shape(realizations_Eve),dtype=int)
    """
    prova = np.int32(rand_sample(strategy_alice_one,2*num_games,tolerance))[0,:]
   ii=0
    for aa in range(num_realizations):
        realizations_Alice[aa,0]=prova[ii]
        realizations_Alice[aa,1]=prova[ii+1]
        ii = ii+2"""
    #realizations_Alice[:,0]=np.int32(rand_sample(strategy_alice_one,2*num_games,tolerance))
    #realizations_Alice[:,1] = np.int32(rand_sample(strategy_alice_one,num_games,tolerance))
    num_mis_det = 0
    dist =0
    prev_Alice = realizations_Alice[0][0]
    for ii in range(num_realizations):
        current_pos_Alice = realizations_Alice[ii][0]
        current_pos_Alice_1 = realizations_Alice[ii][1]
        curr_channel_Alice = ch[0,current_pos_Alice]
        curr_channel_Alice_1 = ch[0,current_pos_Alice_1]
        current_pos_Eve = realizations_Eve[ii][0]
        current_pos_Eve_1 = realizations_Eve[ii][1]
        att_eve = ch[0,current_pos_Eve]
        att_eve_1 = ch[0,current_pos_Eve_1]
        curr_channel_Eve = att_eve+np.random.normal(np.zeros((1,1)),att_eve/np.sqrt(k),None)
        curr_channel_Eve_1 = att_eve_1+np.random.normal(np.zeros((1,1)),att_eve_1/np.sqrt(k),None)

        d= (np.float64(x1[0,current_pos_Alice])-np.float64(x1[0,prev_Alice]))**2+(np.float64(x2[0,current_pos_Alice])-np.float64(x2[0,prev_Alice]))**2
        d_1= (np.float64(x1[0,current_pos_Alice_1])-np.float64(x1[0,current_pos_Alice]))**2+(np.float64(x2[0,current_pos_Alice_1])-np.float64(x2[0,current_pos_Alice]))**2
        dist = dist+np.sqrt(d_1)+np.sqrt(d)
        prev_Alice = current_pos_Alice_1

        test = (k*(curr_channel_Eve-curr_channel_Alice)**2)/(curr_channel_Alice**2)+(k*(curr_channel_Eve_1-curr_channel_Alice_1)**2)/(curr_channel_Alice_1**2)
        #test = (k*(att_eve_fake-curr_channel_Alice)**2)/(curr_channel_Alice**2)
        if test<varphiprime:
            num_mis_det = num_mis_det+1

    p_md = num_mis_det/realizations_Alice.shape[0]
    avg_distance = dist/(2*num_realizations)
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
def find_opt_nes_second(A,D,value_g,tolerance,nes):
    nes = nes.transpose()
    r=A.shape[0]
    c = A.shape[1]
    b = -(-value_g-tolerance)*np.ones((c,1))
    opt_nes = np.zeros((r,r))
    D_prime = np.zeros(np.shape(D))
    for ii in range(r):
        for jj in range(r):
            cost = D[jj,:]@nes[:,jj]
            D_prime[ii][jj]=D[ii][jj]+cost
            """"
            print("Cost:")
            print(cost)
            print("D")
            print(D[ii][jj])"""
    for ii in range(r):
        f= D_prime[ii,:].transpose()
        Aeq = np.ones((1,r))
        lb = np.zeros((r,1))
        ub = np.ones((r,1))
        beq =1
        [local_ne,flag] = greedyStrategy(A,b,Aeq,beq,f,lb,ub)
        opt_nes[:,ii]=local_ne
    return opt_nes


def objective(x, D):
    temp=np.matmul(x.transpose(),D)
    return np.matmul(temp,x)
def quadhess(x,D,a):
    return 2*D
def grad(x,D):
    return 2*np.matmul(D,x)




# START OF SCRIPT

initial_position = 0
print("Terminal (multiple):")
print(initial_position)
n_ch_realization = 50
p_fa = 1e-3 #np.ndarray.tolist(np.linspace(1e-3,1.3e-3,10)) #[1e-5,1.05e-5,1.1e-5,1.15e-5,1.2e-5,1.25e-5,1.3e-5]
dist = 50
ptax = [6,8,10,12,16]
n_ptax= len(ptax)
results_valgame = np.zeros((1,n_ptax))
results_risparmio = np.zeros((1,n_ptax))
results_dist_opt_global = np.zeros((1,n_ptax))
results_dist_opt_approx = np.zeros((1,n_ptax))
num_realizations = np.int32(5e4)
tolerance = 1e-4
strategy=False #true for one opt
k=50
N=2
sigmas =[10]
index_sigma =0
for sigma in sigmas:
    index_ptax = 0
    for p_tax in ptax:
        ch_real = np.arange(0,n_ch_realization,1, dtype=int)
        val_games = np.zeros((1,n_ch_realization), dtype=float)
        risparmio =np.zeros((1,n_ch_realization), dtype=float)
        avg_dist_opt_global = np.zeros((1,n_ch_realization), dtype=float)
        avg_dist_opt_approx = np.zeros((1,n_ch_realization), dtype=float)
        for kk in ch_real:
            filename = './data_from_python/ptax/data_'+str(p_tax)+'_'+str(dist)+'_'+str(sigma)+'_'+str(kk+initial_position)+'.mat'
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
            #prova = mix_row.transpose()@p_md@mix_col
            """ print("qua")
             permutation_col = randperm(size(p_md,2));
            permutation_row = randperm(size(p_md,1));
            [~,inv_perm_col] = sort(permutation_col);
            [~,inv_perm_row] = sort(permutation_row);
            p_md_perm = p_md(:,permutation_col);
            p_md_perm = p_md_perm(permutation_row,: );
            mix_col=mix_col_perm(inv_perm_col); %Perm
            mix_row = mix_row_perm(inv_perm_row); %Perm
            mix_col(mix_col<tolerance)=0;
            """
            mix_row[mix_row<tolerance]=0
            mix_col[mix_col<tolerance]=0
            
            # Distance           
            strategy_eve_naive = np.matlib.repmat(mix_row.transpose(),mix_row.shape[0],1) #repeted by rows
            strategy_Alice_naive = np.matlib.repmat(mix_col.transpose(),mix_col.shape[0],1) #repeted by rows
            #Multiple games
            #p_md_estim_multiple,p_fa_estim_multiple=playGame_multiple(num_realizations,ch,strategy_Alice_naive,strategy_eve_naive,x1_line,x2_line,tolerance,k,varphi_prime_multiple,N)
            val_games[0,kk-1]=val_game
            #Solve strategy
            strategy_Alice_one = find_opt_nes(p_md,dist_matrix,val_game,tolerance).transpose()
            strategy_Alice_second = find_opt_nes_second(p_md,dist_matrix,val_game,tolerance,strategy_Alice_one).transpose()
                #strategy_Alice_check = strategy_Alice.transpose()
            [p_md_estim_global,avg_distance_estim_global] = playGame_multiple_dist(num_realizations,ch,strategy_Alice_one,strategy_Alice_second,strategy_eve_naive,x1_line,x2_line,tolerance,k,varphi_prime_multiple)
            #[p_md_estim_approx,avg_distance_estim_approx] = playGame_multiple_dist(num_realizations,ch,strategy_Alice_one,strategy_Alice_one,strategy_eve_naive,x1_line,x2_line,tolerance,k,varphi_prime_multiple)
            #[p_md_estim_single,avg_distance_estim,avg_distance_naive,gain] = playGame(num_realizations,ch,strategy_Alice_one,strategy_eve_naive,x1_line,x2_line,tolerance,k,varphi_prime,strategy_Alice_naive)
            #avg_dist_naive = mix_col@dist_matrix@mix_col
            #print(((avg_dist_naive-avg_distance_estim)/avg_dist_naive)*100)
            #risparmio[0][kk-1] = ((avg_dist_naive-avg_distance_estim)/avg_dist_naive)*100
            avg_dist_opt_global[0][kk-1] = avg_distance_estim_global
            avg_dist_opt_approx[0][kk-1] = -1
        results_dist_opt_global[index_sigma,index_ptax]=avg_dist_opt_global.mean()
        print(avg_dist_opt_global.mean())
        results_dist_opt_approx[index_sigma,index_ptax]=avg_dist_opt_approx.mean()
        if strategy==True:
            filename = "results_single_opt_prime_les_tol.mat"
            #sp.io.savemat(filename, dict(values_game=results_valgame,risparmio_sing=results_risparmio,dist=distances,p_fas=p_fas, strategy_sing = strategy_Alice, x1_line = x1_line,x2_line=x2_line))
        else:
            filename = "results_multi_opt_optimals"+str(initial_position)+".mat"
            dictionary = {"results_dist_opt_approx"+str(initial_position):results_dist_opt_approx, "results_dist_opt_global_"+str(initial_position):results_dist_opt_global}
            #sp.io.savemat(filename, dict(values_game=results_valgame,results_risparmio=results_risparmio_dic,dist=distances,p_fas=p_fas, strategy_multi= strategy_Alice, x1_line = x1_line,x2_line=x2_line,results_dist_opt_approx = results_dist_opt_approx, results_dist_opt_global=results_dist_opt_global))
            sp.io.savemat(filename,dictionary) 
        index_ptax=index_ptax+1
        print(index_ptax)
    index_sigma=index_sigma+1

"""""
fig, ax = plt.subplots()


ax.semilogy(distances, results_valgame[0,:], label="pfa=1e-4")
ax.semilogy(distances, results_valgame[1,:], label="pfa=5e-4")
ax.semilogy(distances, results_valgame[2,:], label="pfa=1e-3")

plt.xlabel('Distance') 
plt.ylabel('Pmd') 
plt.title('Pmd VS Distance') 
plt.grid(True) 
plt.legend(loc="upper right")
plt.show()


fig, ax = plt.subplots()


ax.plot(p_fas, results_valgame[0,:], label="pfa=1e-4")
ax.plot(p_fas, results_valgame[1,:], label="pfa=5e-4")
ax.plot(p_fas, results_valgame[2,:], label="pfa=1e-3")

plt.xlabel('Distance [m]') 
plt.ylabel('Pmd') 
plt.title('Pmd VS Distance') 
plt.grid(True) 
plt.legend(loc="upper left")
plt.show()

fig, ax = plt.subplots()


ax.plot(p_fas, results_risparmio[0,:], label="dist=10")
#ax.plot(p_fas, results_risparmio[1,:], label="dist=50")
#ax.plot(p_fas, results_risparmio[2,:], label="dist=100")

plt.xlabel('Distance') 
plt.ylabel('Risparmio %') 
plt.title('Risparmio VS Distance') 
plt.grid(True) 
plt.legend(loc="upper right")
plt.show()

print("Fine")
"""""

