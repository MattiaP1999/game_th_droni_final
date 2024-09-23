close all
clear all
load("./data/Results_literature.mat")

%results_optimal=[0.019326, 0.015345, 0.015712, 0.014083, 0.012593, 0.010879,0.009166, 0.007999];
figure
loglog(p_fas,results_uni_smart,'-o')
hold on
loglog(p_fas,results_uni_uni,'-o')
loglog(p_fas,results_optimal,'-o')
ylim([0.005,1])
grid on
set(gca,'TickLabelInterpreter','latex')
ylabel("$P_{\rm md}$",'interpreter','latex')
xlabel("$P_{\rm fa}$",'interpreter','latex')
%title("Heuristic VS Game theoretical solutions",'interpreter','latex')
legend("Unfiorm-Smart",'Uniform-Uniform','Game Theory','Location','best')

