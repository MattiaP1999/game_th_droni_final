close all
clear all
load("./data/distance_altezza/processed_data.mat")
% % 
altezza = [10,20,30,50,70,100];
up_naive = mean_naive+sqrt(var_naive);
down_naive = mean_naive-sqrt(var_naive);

up_opt = mean_one_opt+sqrt(var_one_opt);
down_opt = mean_one_opt-sqrt(var_one_opt);

up_multi = mean_multi+sqrt(var_multi);
down_multi = mean_multi-sqrt(var_multi);


figure
set(gca,'TickLabelInterpreter','latex')
plot(altezza,mean_naive,'blue')
hold on
plot(altezza,mean_one_opt,"-^")
plot(altezza,mean_multi,"-o")
grid on
ylabel("Average distance [m]",'interpreter','latex')
xlabel('$h$','interpreter','latex')
%legend('LP solution','SNE','MNE','interpreter','latex','Location','best')
patch([altezza fliplr(altezza)],[up_naive fliplr(down_naive)],'blue',"FaceAlpha",0.2,'LineStyle','none')
patch([altezza fliplr(altezza)],[up_opt fliplr(down_opt)],'red',"FaceAlpha",0.2,'LineStyle','none')
patch([altezza fliplr(altezza)],[up_multi fliplr(down_multi)],'yellow',"FaceAlpha",0.2,'LineStyle','none')


% 
% figure
% set(gca,'TickLabelInterpreter','latex')
% plot(altezza,average_valgame,'-o')
% grid on
% ylabel("$P_{\rm md}$",'interpreter','latex')
% xlabel('h','interpreter','latex')
% legend("$P_{\rm fa}=10^{-3}$",'interpreter','latex','Location','best')