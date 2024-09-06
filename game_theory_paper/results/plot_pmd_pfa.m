close all
clear all
%load("./data/pmd_pfa_rounds/results_pmd_pfa_less.mat")
%load("./data/Results_pfa_pmd_round_1.mat")
load("./data/multi_round/Results_pfa_pmd_round_1.mat")
load("./data/multi_round/Results_pfa_pmd_round_2.mat")
load("./data/multi_round/Results_pfa_pmd_round_4.mat")

figure
loglog(p_fas,values_game_1(1,:));
hold on
%loglog(p_fas,values_game_1(2,:));
%loglog(p_fas,values_game_1(3,:));
loglog(p_fas,values_game_2(1,:),'-x');
%loglog(p_fas,values_game_2(2,:),'-x');
%loglog(p_fas,values_game_2(3,:),'-x');
loglog(p_fas,values_game_4(1,:),'-o');
%loglog(p_fas,values_game_4(2,:),'-o');
%loglog(p_fas,values_game_4(3,:),'-o');
grid on
set(gca,'TickLabelInterpreter','latex')
ylabel("$P_{\rm md}$",'interpreter','latex')
xlabel("$P_{\rm fa}$",'interpreter','latex')
title("$P_{\rm md}$ vs $P_{\rm fa}$, at different showing variance $\sigma_{\rm s}$",'interpreter','latex')
legend('$\sigma_{\rm s}= 3, N=1$','$\sigma_{\rm s}= 10,N=1$','$\sigma_{\rm s}: 16,N=1$','$\sigma_{\rm s}= 3, N=2$','$\sigma_{\rm s}= 10,N=2$','$\sigma_{\rm s}: 16,N=2$','$\sigma_{\rm s}= 3, N=4$','$\sigma_{\rm s}= 10,N=4$','$\sigma_{\rm s}: 16,N=4$','interpreter','latex',Location='best')
%legend('sigma: 3','sigma: 10','sigma: 16',"simulation")
% axes('position',[.59 .177 .30 .30])
% box on % put box around new pair of axes
% indexOfInterest = (p_fas > 0.01); % range of t near perturbation
% loglog(p_fas(indexOfInterest),values_game_1(1,indexOfInterest))
% hold on
% grid on
% plot(p_fas(indexOfInterest),values_game_1(2,indexOfInterest))
% plot(p_fas(indexOfInterest),values_game_1(3,indexOfInterest))
% axis tight