close all
clear all
%load("./data/results_pmd_pfa.mat")
load("./data/results_pmd_pfa_zoom.mat")
p_fas = p_fas_zoom;
results_valgame_sim = results_valgame_sim_zoom;
values_game = values_game_zoom;
% x_axes_1 = p_fas(:);
% x_axes = [x_axes_1,p_fas_zoom];
% values_game_1_p = values_game_1(:,:);
% values_game_1 = [values_game_1_p,values_game_zoom];
% 
% results_valgame_sim_1_p = results_valgame_sim_1(:,:);
% results_valgame_sim_1 = [results_valgame_sim_1_p,results_valgame_sim_zoom];

figure
loglog(p_fas,values_game(1,:));
hold on
loglog(p_fas,values_game(2,:));
loglog(p_fas,values_game(3,:));
loglog(p_fas,results_valgame_sim(1,:),"x");
loglog(p_fas,results_valgame_sim(2,:),"x");
loglog(p_fas,results_valgame_sim(3,:),"x");
grid on
set(gca,'TickLabelInterpreter','latex')
ylabel("$P_{\rm md}$",'interpreter','latex')
xlabel("$P_{\rm fa}$",'interpreter','latex')

legend('$\sigma_{\rm s}: 10$','$\sigma_{\rm s}: 13$','$\sigma_{\rm s}: 16$',"simulated values",'interpreter','latex','Location','best')
matlab2tikz('myfile.tex');
% axes('position',[.20 .175 .25 .25])
% box on % put box around new pair of axes
% indexOfInterest = (x_axes >= 1e-6); % range of t near perturbation
% loglog(x_axes(indexOfInterest),values_game_1(1,indexOfInterest))
% hold on
% grid on
% loglog(x_axes(indexOfInterest),values_game_1(2,indexOfInterest))
% loglog(x_axes(indexOfInterest),values_game_1(3,indexOfInterest))
% loglog(x_axes(indexOfInterest),results_valgame_sim_1(1,indexOfInterest),"x")
% loglog(x_axes(indexOfInterest),results_valgame_sim_1(2,indexOfInterest),"x")
% loglog(x_axes(indexOfInterest),results_valgame_sim_1(3,indexOfInterest),"x")
% axis tight