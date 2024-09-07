close all
clear all
load("./data/distance_ptx/workspace.mat")
ptax = [4,6,8,10,12,16];
ptax = ptax.^2;
dist_naive = (results_dist_naive_0_multi+results_dist_naive_0)/2;
figure
plot(ptax,dist_naive);
hold on
plot(ptax,results_dist_opt_0,'--o');
plot(ptax,results_dist_opt_0_multi,'-o');
%semilogy(ptax,value_games(3,:),'-o');
grid on
xlim([16 256])
set(gca,'TickLabelInterpreter','latex')
ylabel("Average distance",'interpreter','latex')
xlabel("Number of grid points",'interpreter','latex')
legend('LP','SNE','MNE','interpreter','latex')