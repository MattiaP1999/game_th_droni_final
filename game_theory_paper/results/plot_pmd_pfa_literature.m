close all
clear all
load("./data/Results_literature.mat")

figure
loglog(p_fas,results_uni_smart,'-o')
hold on
loglog(p_fas,result_k_double,'-o')
loglog(p_fas,results_optimal,'-o')
ylim([0.005,1])
grid on
set(gca,'TickLabelInterpreter','latex')
ylabel("$P_{\rm md}$",'interpreter','latex')
xlabel("$P_{\rm fa}$",'interpreter','latex')
%title("Heuristic VS Game theoretical solutions",'interpreter','latex')
legend("Globecom Smart",'Globecom','Game Theory')
%matlab2tikz('myfile.tex');

