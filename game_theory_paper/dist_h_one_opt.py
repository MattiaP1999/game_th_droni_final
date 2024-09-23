import numpy as np
import scipy as sp
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

def objective(x, D):
    temp=np.matmul(x.transpose(),D)
    return np.matmul(temp,x)
def quadhess(x,D,a):
    return 2*D
def grad(x,D):
    return 2*np.matmul(D,x)
def getVariance(mean,vector,N):
    var = 0
    for el in vector[0,:]:
        var = var + (el-mean)**2
    return var/N



# START OF SCRIPT

initial_position = 150
print("Terminal (single):")
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

            strategy_eve_naive = np.matlib.repmat(mix_row.transpose(),mix_row.shape[0],1) #repeted by rows
            strategy_Alice_naive = np.matlib.repmat(mix_col.transpose(),mix_col.shape[0],1) #repeted by rows
            val_games[0,kk-1]=val_game
            #Solve strategy
            global_opt_ne = game_solver_one_opt(p_md,dist_matrix,val_game,tolerance)
            strategy_Alice = np.matlib.repmat(global_opt_ne.transpose(),global_opt_ne.shape[0],1)
            avg_distance_estim = global_opt_ne@dist_matrix@global_opt_ne
            avg_distance_naive = mix_col@dist_matrix@mix_col
            avg_dist_naive[0][kk-1] = avg_distance_naive
            avg_dist_opt[0][kk-1] = avg_distance_estim
        results_valgame[index_sigma,distance_index]=val_games.mean()
        results_dist_naive[index_sigma,distance_index]=avg_dist_naive.mean()
        results_dist_opt[index_sigma,distance_index]=avg_dist_opt.mean()
        #Variance
        results_variance_naive[index_sigma,distance_index]=getVariance(mean_naive[distance_index],avg_dist_naive,n_ch_realization)
        results_variance_opt[index_sigma,distance_index]=getVariance(mean_opt_one[distance_index],avg_dist_opt,n_ch_realization)

        filename = "results_single_opt_dist_variance"+str(initial_position)+".mat"
        dictionary = {"values_game_"+str(initial_position):results_valgame, "results_dist_opt_"+str(initial_position):results_dist_opt, "results_dist_naive_"+str(initial_position):results_dist_naive,"results_variance_naive_"+str(initial_position):results_variance_naive,"results_variance_opt_"+str(initial_position):results_variance_opt}
        sp.io.savemat(filename,dictionary) 
        print(distance_index) 
        distance_index=distance_index+1
    index_sigma=index_sigma+1