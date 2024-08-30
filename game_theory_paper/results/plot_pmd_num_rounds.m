close all
clear all
%load("./data/Results_num_rounds_complete.mat")
load("./data/Results_num_rounds.mat")
x_axes = [1,2,3,4,5,6];
figure
semilogy(x_axes,values_game(1,:),'-o')
hold on
semilogy(x_axes,values_game(2,:),'-o')
semilogy(x_axes,values_game(3,:),'-o')
grid on
xticks(1:1:6)

set(gca,'TickLabelInterpreter','latex')
ylabel("$P_{\rm md}$",'interpreter','latex')
xlabel("$N$",'interpreter','latex')
legend('$P_{\rm fa}: 10^{-4}$','$P_{\rm fa}: 10^{-3}$','$P_{\rm fa}: 10^{-2}$','interpreter','latex')