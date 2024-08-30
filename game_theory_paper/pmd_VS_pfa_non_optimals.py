import numpy as np
import scipy as sp
import os
import numpy.matlib
import matplotlib as plt

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def remove_diag(x):
    x_no_diag = np.ndarray.flatten(x)
    x_no_diag = np.delete(x_no_diag, range(0, len(x_no_diag), len(x) + 1), 0)
    x_no_diag = x_no_diag.reshape(len(x), len(x) - 1)
    return x_no_diag
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
    try:
        c = A.shape[1]
    except:
        print("exception occured, handled")
        c=1
    AA = np.ones((c,r+1))
    AA[0:c,0:r] = -A.transpose()
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

def objective(x, D):
    temp=np.matmul(x.transpose(),D)
    return np.matmul(temp,x)
def quadhess(x,D,a):
    return 2*D
def grad(x,D):
    return 2*np.matmul(D,x)
def oneOpt(A_ub,b_ub,A_eq,b_eq,D,lb,ub):
    bounds=np.array([lb, ub]).transpose()
    eq_constraint=sp.optimize.LinearConstraint(A=A_eq,lb=b_eq,ub=b_eq,keep_feasible=False)
    neq_constraint = sp.optimize.LinearConstraint(A_ub,-np.inf*np.ones(np.array(ub).shape),b_ub)

    termination_f = -1
    n_att = 1
    num_multi_start = 20
    best_obj_val = np.inf
    at_least_one = False
    while n_att<=num_multi_start or at_least_one==False:
        x0=np.random.rand(225,)
        x0=x0/np.sum(x0)
        res=sp.optimize.minimize(objective, x0=x0, args=(D), method=None, jac=None, hessp=None, bounds=bounds, constraints=(eq_constraint,neq_constraint), tol=1e-5, callback=None, options=None)
        termination_f = res['status']
        if termination_f==0:
            at_least_one = True
            obj_val = res['fun']
            if obj_val<best_obj_val:
                local_ne = res['x']
                best_obj_val = obj_val
        n_att = n_att+1
    return local_ne
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





# START OF SCRIPT
n_ch_realization = 10
#p_fas = [1e-2]
n_pfas = 10
p_fas = np.linspace(1e-3,1e-1,n_pfas)
sigmas = [10,13,16]
results_Trudy_forte = np.zeros((3,n_pfas))
results_uniforme = np.zeros((3,n_pfas))
results_theoretic = np.zeros((3,n_pfas))
num_realizations = np.int32(1e5)
print("qua")
tolerance = 1e-5
strategy=True #true for one opt
k=50
distance =50
counter =0
index_sigma =0
ptax = 15
for sigma in sigmas:
    pfa_index = 0
    for pfa in p_fas:
        ch_real = np.arange(0,n_ch_realization,1, dtype=int)
        val_games_Trudy_forte = np.zeros((1,n_ch_realization), dtype=float)
        val_games_uniforme = np.zeros((1,n_ch_realization), dtype=float)
        val_games_theoretic = np.zeros((1,n_ch_realization), dtype=float)
        for kk in ch_real:
            filename = './data_from_python/complete_dataset/data_'+str(ptax)+'_'+str(distance)+'_'+str(sigma)+'_'+str(kk)+'.mat'
            mat_f_load = sp.io.loadmat(filename)
            #mat_f_load = sp.io.loadmat('./data_from_python/dist/data_gth_pythontemp.mat') 
            ch = (10**(-np.array(mat_f_load["ch"])/20))**2
            ch = ch+min(min(ch))/10
            x1 = np.array(mat_f_load["x1"])
            x2 = np.array(mat_f_load["x2"])
            n_att = ch.shape[1]
            x1_line = np.reshape(x1,(1,-1))
            x2_line = np.reshape(x2,(1,-1))
            dist_matrix = np.reshape(get_distance(x1_line,x2_line),[x1_line.size,x1_line.size])
            varphi_prime= sp.stats.chi2.ppf(1-pfa, 1, loc=0, scale=1)
            #gi_star= np.zeros((n_att,n_att))
            p_md = np.zeros((n_att,n_att)) 
            for ii in range(n_att):
                for jj in range(n_att):
                    nc = ((ch[0][ii]-ch[0][jj])*np.sqrt(k)/ch[0][ii])**2
                    x=varphi_prime*(ch[0][jj]/ch[0][ii])**2
                    p_md[ii,jj] = sp.stats.ncx2.cdf(x, 1, nc, loc=0, scale=1)
            p_md[p_md<tolerance]=0
            p_md[p_md<tolerance]=0
            val_game,mix_row = game_solver(p_md)
            num_col=p_md.shape[1]
            #Check of the game
            mix_col_fake = np.ones(num_col)/num_col
            val_game_uniforme_k = np.max(mix_col_fake.transpose()@p_md.transpose())
            max_pmd = np.max(p_md)
            val_games_uniforme[0][kk-1]=val_game_uniforme_k
            val_games_Trudy_forte[0][kk-1]=max_pmd
            val_games_theoretic[0][kk-1]=val_game
        #results_valgame[index_pfa,index_dist]=val_games.mean()
        #results_risparmio[index_pfa,index_dist]=risparmio.mean()
        results_Trudy_forte[index_sigma,pfa_index]=val_games_Trudy_forte.mean()
        results_uniforme[index_sigma,pfa_index] = val_games_uniforme.mean()
        results_theoretic[index_sigma,pfa_index] = val_games_theoretic.mean() 
        counter = counter +1
        print(str(counter/np.float16(results_Trudy_forte.size)*100))
        pfa_index=pfa_index+1
    index_sigma=index_sigma+1
sp.io.savemat("Results_pmd_pfa_non_opt_complete.mat", dict(results_Trudy_forte=results_Trudy_forte,results_uniforme=results_uniforme,p_fas=p_fas,val_game = results_theoretic))
""""
    # plotting
fig, ax = plt.subplots()



ax.semilogy(p_fas, results_uniforme[0,:], label="sigma=3, uniforme")
ax.semilogy(p_fas, results_uniforme[1,:], label="sigma=10, uniforme")
ax.semilogy(p_fas, results_uniforme[2,:], label="sigma=16,uniforme")
ax.semilogy(p_fas, val_games_Trudy_forte[0,:],linestyle='--')
ax.semilogy(p_fas, val_games_Trudy_forte[1,:],linestyle='--')
ax.semilogy(p_fas, val_games_Trudy_forte[2,:],linestyle='--')
print("Stop")

plt.xlabel('Pfa') 
plt.ylabel('Pmd') 
plt.title('Pfa VS Pmd') 
plt.grid(True) 
plt.legend(loc="upper right")
plt.show()
"""



