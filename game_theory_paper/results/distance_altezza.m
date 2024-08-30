close all
clear all
load("./data/distance_altezza/average_multi.mat")
load("./data/distance_altezza/average_single.mat")

% 
figure
set(gca,'TickLabelInterpreter','latex')
plot(altezza,average_res_naive_single)
hold on
plot(altezza,average_res_opt_single,"-^")
plot(altezza,average_res_opt_multi,"-o")
grid on
ylabel("Average distance",'interpreter','latex')
xlabel('$h$','interpreter','latex')
legend('LP solution','SNE','MNE','interpreter','latex','Location','best')
% 
figure
set(gca,'TickLabelInterpreter','latex')
plot(altezza,average_valgame,'-o')
grid on
ylabel("$P_{\rm md}$",'interpreter','latex')
xlabel('h','interpreter','latex')
legend("$P_{\rm fa}=10^{-3}$",'interpreter','latex','Location','best')