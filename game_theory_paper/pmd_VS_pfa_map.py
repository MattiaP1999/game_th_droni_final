import numpy as np
import scipy as sp
import os
import numpy.matlib
import matplotlib as plt
import random as rn
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



# START OF SCRIPT
n_ch_realization = 50
#p_fas = [1e-2]
n_pfas = 10
p_fas = np.logspace(-3,-1,n_pfas,base = 10)
sigmas = [10]
results_Trudy_random = np.zeros((1,n_pfas))
results_diagonal = np.zeros((1,n_pfas))
results_normal = np.zeros((1,n_pfas))
num_realizations = np.int32(1e5)
tolerance = 1e-5
k=50
distance =50
counter =0
index_sigma =0
ptax = 8
for sigma in sigmas:
    pfa_index = 0
    for pfa in p_fas:
        ch_real = np.arange(0,n_ch_realization,1, dtype=int)
        val_games_Trudy_random = np.zeros((1,n_ch_realization), dtype=float)
        val_games_diagonal = np.zeros((1,n_ch_realization), dtype=float)
        val_games_normal= np.zeros((1,n_ch_realization), dtype=float)
        for kk in ch_real:
            filename = './data_from_python/ptax/data_'+str(ptax)+'_'+str(distance)+'_'+str(sigma)+'_'+str(kk)+'.mat'
            mat_f_load = sp.io.loadmat(filename)
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
            val_game_half = 0
            num_rand = 100
            val_game,mix_row = game_solver(p_md)
            for i in range(num_rand):
                indexes_del = rn.sample(range(n_att), np.int32(np.floor(n_att*0.5)))
                p_md_half = numpy.delete(p_md, indexes_del, axis=1)
                val_game_half_temp,mix_row = game_solver(p_md_half)
                val_game_half = val_game_half + val_game_half_temp
            val_game_half = val_game_half/num_rand
            p_md_diag = p_md.transpose()
            p_md_diag=p_md_diag[~np.eye(p_md_diag.shape[0],dtype=bool)].reshape(p_md_diag.shape[0],-1)
            p_md_diag = p_md_diag.transpose()
            val_game_diag,mix_row = game_solver(p_md_diag)
            #Check of the game
            val_games_Trudy_random[0][kk-1]=val_game_half
            val_games_diagonal[0][kk-1]=val_game_diag
            val_games_normal[0][kk-1]=val_game
        results_Trudy_random[index_sigma,pfa_index]=val_games_Trudy_random.mean()
        results_diagonal[index_sigma,pfa_index] = val_games_diagonal.mean()
        results_normal[index_sigma,pfa_index] = val_games_normal.mean() 
        counter = counter +1
        pfa_index=pfa_index+1
        print(pfa_index/n_pfas*100)
        filename = "Results_pmd_pfa_map_"+str(ptax)+".mat" #ATTENZIONE
        sp.io.savemat(filename, dict(results_Trudy_random_8=results_Trudy_random,results_diagonal_8=results_diagonal,p_fas=p_fas,val_games_normal_8 = results_normal))
    index_sigma=index_sigma+1


