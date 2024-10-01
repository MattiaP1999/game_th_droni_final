close all
clear all
load("./data/distance_ptx/workspace.mat")
load("./data/distance_ptx/results_multi_opt_optimals0.mat")

ptax = [4,6,8,10,12,16];
ptax = ptax.^2;
dist_naive = (results_dist_naive_0_multi+results_dist_naive_0)/2;
figure
plot(ptax,2*dist_naive,'-x');
hold on
plot(ptax,2*results_dist_opt_0,'-o');
plot(ptax,2*results_dist_opt_approx0,'-o');
plot(ptax,2*results_dist_opt_global_0,'--o');
%semilogy(ptax,value_games(3,:),'-o');
grid on
xlim([16 256])
xlabel(ptax)
set(gca,'TickLabelInterpreter','latex')
ylabel("Average distance",'interpreter','latex')
xlabel("Number of grid points",'interpreter','latex')
legend('LP','SNE','MNE','Global Opt','interpreter','latex')
matlab2tikz('myfile.tex');