close all
clear all
load("./data/Results_pmd_pfa_non_opt_complete.mat")

x_axes = p_fas;
figure
loglog(x_axes,results_Trudy_forte(1,:))
hold on
loglog(x_axes,results_uniforme(1,:),"--")
loglog(x_axes,results_uniforme(2,:),"--")
loglog(x_axes,results_uniforme(3,:),"--")
loglog(x_axes,val_game(1,:),"-x")
loglog(x_axes,val_game(2,:),"-x")
loglog(x_axes,val_game(3,:),"-x")
grid on
set(gca,'TickLabelInterpreter','latex')
ylabel("$P_{\rm md}$",'interpreter','latex')
xlabel("$P_{\rm fa}$",'interpreter','latex')
title("Heuristic VS Game theoretical solutions",'interpreter','latex')
legend("Degenerate",'Uniform, $\sigma_{\rm s}= 10$','Uniform, $\sigma_{\rm s}= 13$','Uniform, $\sigma_{\rm s}: 16$','Game theory, $\sigma_{\rm s}= 10$','Game theory, $\sigma_{\rm s}= 13$','Game theory, $\sigma_{\rm s}: 16$','interpreter','latex')