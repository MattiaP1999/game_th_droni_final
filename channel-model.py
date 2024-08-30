#%matplotlib widget
import warnings
warnings.filterwarnings('ignore')
from math import exp, floor, ceil, sqrt, log10
import numpy as np
from scipy import constants
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d
import os


alfa=2 # coefficiente di path-loss equazione di friis
rx_dists=[10,20,30,40,50,60,70,80,90,100]#[5,15]#10 #distanza ricevitore-centro dello spazio in metri (altezza de drone)
MHz=2400
f=MHz*(10**6) # frequenza in Hz
wavelength = constants.speed_of_light/f
#COHD=wavelength*10 #1.67
COHD=10 # coherence distance shadowing in wavelength
SIGMAX=[10,13,16]#[3,10,16] # 6 varianza dello shadowing
PTAX=[15]#15# Nel libro è N1, oppure N2; qui si considera uno spazio "quadrato". E' il numero di punti per ogni asse della griglia
LENGTH = 420
#STEP=30 # passo della griglia (distanza tra i punti, in metri) redefined below

NUM_SHAD = 100 #number of shadowing realizations
########################################################################################
'''
Funzione di correlazione tra due componenti di shadowing di due ricevitori posti a distanza delta
(MODELLO GUDMUNDSON)
'''

def corr_xi(sigma_xi, delta, coh_dist):
    return (sigma_xi**2)*exp(-(abs(delta))/coh_dist)

corr_xi = np.frompyfunc(corr_xi, 3, 1)

########################################################################################

'''
funzione di calcolo del path loss
'''    
def path_loss(alfa, km_dist, MHz_freq):
    return 32.4 + alfa*10*log10(km_dist) + 20*log10(MHz_freq)

path_loss_matrix = np.frompyfunc(path_loss, 3, 1)


########################################################################################
'''
Genera le coordinate dello spazio (griglia)
'''

def create_space(pt_ax, step_size):
    

    #crea punti del piano
    c1 = step_size*(np.arange(-floor(pt_ax/2), ceil(pt_ax/2)))
    c2 = step_size*(np.arange(-floor(pt_ax/2), ceil(pt_ax/2)))
    
    #crea matrice di cordinate dai vettori c1 e c2
    return np.meshgrid(c1,c2)


########################################################################################
'''
Crea il filtro (H nel modello, riferimento in Algorithms for Communications Systems and their Applications, Cap. 4 di Nevio Benvenuto, Giovanni Cherubini, Stefano Tomasin)
'''

def create_filter(cc1, cc2, sigma_xi, coh_dist):
    
    #calcola matrice delle distanze di ogni punto dall'origine
    dd = np.sqrt(cc1**2+cc2**2)
    
    #calcola la correlazione tra le componenti di shadowing all'origine e al punto a distanza delta
    rr_d = np.float64(corr_xi(sigma_xi, dd, coh_dist))
    
    #calcola DFT2 della matrice di correlazione dei delta
    Pk = sp.fft.fft2(rr_d)
    
    '''
    Costruzione filtro
    '''
    KHk = np.sqrt(Pk) # filtro non normalizzato, manca K
    hn= sp.fft.ifft2(KHk)
    
    hn_energy = sum(sum(abs(hn)**2)) # calcolo l'energia di hn
    
    if(hn_energy>0):
        hn=hn/np.sqrt(hn_energy) # normalizzazione
        
    return hn

########################################################################################


'''
Generazione rumore gaussiano sui punti dello spazio
'''
def generate_gaussian_noise(pt_ax, sigma_xi):

    n=(pt_ax)**2 #numero valori gaussiani da generare
    w = np.random.normal(loc=0, scale=sigma_xi/2,
                            size=(n, 2)).view(np.complex128)  
    
    return np.reshape(w,(pt_ax,pt_ax)).view(np.complex128) #crea la matrice delle dimensioni adatte

########################################################################################

'''
Simula shadowing data la matrice di variabili aleatorie gaussiane e il filtro (Riferimento nel libro citato sopra)
Metodo di generazione tramite moltiplicazione nel dominio della frequenza
'''
def compute_shadowing(hn, w):
    
    Hn = sp.fft.fft2(hn)
    return sp.fft.ifft2(Hn*w)

########################################################################################

'''
Simula shadowing come sopra ma tramite convoluzione 2D (non passa per il dominio della frequenza)
'''
def compute_shadowing_convolved(hn, w):
    return sp.signal.convolve2d(hn, w, mode='same')

for ptax in PTAX:
    STEP = LENGTH/(ptax-1)
    x1, x2 = create_space(ptax, STEP)
    #hn = create_filter(x1, x2, SIGMAX, COHD)
    for rx_dist in rx_dists:
        for sigma in SIGMAX:
            for i in range(NUM_SHAD):
                '''
                Genera shadowing
                '''
                coh_dist = COHD*wavelength
                #print(coh_dist) 
                hn = create_filter(x1, x2, sigma, coh_dist)
                w = generate_gaussian_noise(ptax, sigma)
                xi = compute_shadowing_convolved(hn, w) #matrice della realizzazione dello shadowing sulla mappa (SOLO SHADOWING, NO PATH LOSS)

                '''
                Calcola matrice del path loss dovuto alla distanza
                '''
                center_d = np.sqrt(x1**2+x2**2) #distanza del drone dal centro in metri
                d_rx_km = ( np.sqrt(center_d**2 + rx_dist**2) ) / 1000# distanza del drone dal ricevitore in kilometri
                pl = np.float64(path_loss_matrix(alfa, d_rx_km, MHz)) #matrice del path loss
                ch = pl+xi; # matrice del canale nello spazio (shadowing + path loss).
                ch = ch.real #rimuove parte complessa spuria (altrimenti genera una parte complessa dai valori infinitesimali, causa i limiti del calcolatore (?) )
                # Valori max e min dell'attenuazione del canale
                pl_max = np.max(np.max(pl))
                ch_max = np.max(np.max(ch))
                gain_min = (10**(-pl_max/20))**2
                gain_min_fake = (10**(-ch_max/20))**2
                #sigma = gain_min/10
                #sigma_fake = gain_min_fake/10

                #m_ok = gain_min_fake+sigma
                #m_fake = gain_min_fake+sigma_fake

                #print(20*np.log10(m_ok/m_fake))

                max_val = np.amax(ch)
                min_val = np.amin(ch)
                #Mattia
                p_mat = []
                ch_squezed = ch.reshape(-1)
                #print(10**(ch/20))
                #print(np.min((10**(ch/20))))
                for el in ch_squezed:
                    p_mat.append(ch_squezed-el)
                p_mat = np.array(p_mat)
                #print(p_mat[12,:])
                filename = os.path.join(os.path.dirname(__file__), 'game_theory_paper\data_from_python\complete_dataset\data')
                sp.io.savemat(filename+"_"+str(ptax)+"_"+str(rx_dist)+"_"+str(sigma)+"_"+str(i+100)+".mat", dict(ch=ch_squezed,x1=x1,x2=x2,p_mat = p_mat))
"""""
                #indici lineari dei valori max e min
                max_index = np.unravel_index(np.argmax(ch), ch.shape)
                min_index = np.unravel_index(np.argmin(ch), ch.shape)
                #indici matriciali dei valori max e min
                max_i = max_index[0]
                max_j = max_index[1]
                min_i = min_index[0]
                min_j = min_index[1]
                #coordinate dei valori max e min
                max_x = x1[max_i, max_j]
                max_y = x2[max_i, max_j]
                min_x = x1[min_i, min_j]
                min_y = x2[min_i, min_j]
                dist = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2) # distanza tra max e min
                fig_map = plt.figure(figsize=(10,8))
                ax_map = fig_map.add_subplot()
                ax_map.set_xlabel("[m]")
                ax_map.set_ylabel("[m]")
                ax_map.set_title("Channel model due to path loss and shadowing")
                cntf_map = ax_map.contourf(x1, x2, ch, levels=1000, cmap=cm.coolwarm)
                cbar_map = fig_map.colorbar(cntf_map)

                # Imposta barra laterale
                tick_positions = np.linspace(cbar_map.vmin, cbar_map.vmax, len(cbar_map.get_ticks()))
                tick_labels = [f"{tick:.3f} dB" for tick in cbar_map.get_ticks()]
                cbar_map.set_ticks(tick_positions)
                cbar_map.set_ticklabels(tick_labels)
                #Salva la figura
                filename = os.path.join(os.path.dirname(__file__), 'game_theory_paper\data_from_python\complete_dataset\images\imag')
                plt.show()
                #plt.savefig(filename+"_"+str(rx_dist)+"_"+str(sigma)+".png")"""


