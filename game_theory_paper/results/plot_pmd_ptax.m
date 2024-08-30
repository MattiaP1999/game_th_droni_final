close all
load("./data/pmd_ptx/results_pmd_ptax_30.mat")
%load("./data/pmd_ptx/results_pmd_ptax_0.mat")

value_games = (values_game_0+values_game_30)/2;
ptax = ptax.^2;
figure
semilogy(ptax,value_games(1,:),'-o');
hold on
semilogy(ptax,value_games(2,:),'-o');
semilogy(ptax,value_games(3,:),'-o');
xlim([0 256])
grid on
set(gca,'TickLabelInterpreter','latex')
ylabel("$P_{\rm md}$",'interpreter','latex')
xlabel("Number of grid points",'interpreter','latex')
legend('$P_{\rm fa}: 10^{-4}$','$P_{\rm fa}: 10^{-3}$','$P_{\rm fa}: 10^{-2}$','interpreter','latex')