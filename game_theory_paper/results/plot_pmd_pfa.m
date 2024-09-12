close all
clear all
%load("./data/pmd_pfa_rounds/results_pmd_pfa_less.mat")
%load("./data/Results_pfa_pmd_round_1.mat")
load("./data/multi_round/Results_pfa_pmd_round_1.mat")
load("./data/multi_round/Results_pfa_pmd_round_2.mat")
load("./data/multi_round/Results_pfa_pmd_round_3.mat")
%
index_to_del = [3,6,7,9,10,11,12,13,15,17,18,19];
values_game_1(:,index_to_del) =[];
values_game_2(:,index_to_del) =[];
values_game_3(:,index_to_del) =[];
p_fas(:,index_to_del) =[];
figure
loglog(p_fas,values_game_1(1,:));
hold on
loglog(p_fas,values_game_1(2,:));
loglog(p_fas,values_game_1(3,:));
loglog(p_fas,values_game_2(1,:),'-x');
loglog(p_fas,values_game_2(2,:),'-x');
loglog(p_fas,values_game_2(3,:),'-x');
loglog(p_fas,values_game_3(1,:),'-o');
loglog(p_fas,values_game_3(2,:),'-o');
loglog(p_fas,values_game_3(3,:),'-o');
grid on
set(gca,'TickLabelInterpreter','latex')
ylabel("$P_{\rm md}$",'interpreter','latex')
xlabel("$P_{\rm fa}$",'interpreter','latex')
legend('$\sigma_{\rm s}= 10, N=1$','$\sigma_{\rm s}= 13,N=1$','$\sigma_{\rm s}: 16,N=1$','$\sigma_{\rm s}= 10, N=2$','$\sigma_{\rm s}= 13,N=2$','$\sigma_{\rm s}: 16,N=2$','$\sigma_{\rm s}= 10, N=3$','$\sigma_{\rm s}= 13,N=3$','$\sigma_{\rm s}: 16,N=3$','interpreter','latex',Location='best')
%legend('sigma: 10','sigma: 13','sigma: 16',"simulation")
% % axes('position',[.59 .177 .30 .30])
% % box on % put box around new pair of axes
% % indexOfInterest = (p_fas > 0.01); % range of t near perturbation
% % loglog(p_fas(indexOfInterest),values_game_1(1,indexOfInterest))
% % hold on
% % grid on
% % plot(p_fas(indexOfInterest),values_game_1(2,indexOfInterest))
% % plot(p_fas(indexOfInterest),values_game_1(3,indexOfInterest))
% % axis tight