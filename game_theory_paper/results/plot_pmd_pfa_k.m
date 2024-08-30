close all
clear all
load("./data/Results_pmd_pfa_k.mat")

figure
loglog(p_fa,values_game(1,:),'-o');
hold on
loglog(p_fa,values_game(2,:),'-o');
loglog(p_fa,values_game(3,:),'-o');
grid on
set(gca,'TickLabelInterpreter','latex')
ylabel("$P_{\rm md}$",'interpreter','latex')
xlabel("$P_{\rm fa}$",'interpreter','latex')
legend('$K= 50$','$K= 70$','$K= 100$','interpreter','latex')