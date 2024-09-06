close all
clear all
load("./data/pmd_map/Results_pmd_pfa_map_6.mat")
load("./data/pmd_map/Results_pmd_pfa_map_8.mat")
figure
loglog(p_fas,val_games_normal_6,'-o')
hold on
loglog(p_fas,results_diagonal_6,'-o')
loglog(p_fas,results_Trudy_random_6,'-o')
loglog(p_fas,val_games_normal_8,'--o')
loglog(p_fas,results_diagonal_8,'--o')
loglog(p_fas,results_Trudy_random_8,'--o')
grid on
set(gca,'TickLabelInterpreter','latex')
ylabel("$P_{\rm md}$",'interpreter','latex')
xlabel("$P_{\rm fa}$",'interpreter','latex')
legend("Normal, grid-points = 36","No-diagonal, grid-points = 36","Half-points, grid-points = 36", ...
    "Normal, grid-points = 64","No-diagonal, grid-points = 64","Half-points, grid-points = 64",'interpreter','latex',Location='best')