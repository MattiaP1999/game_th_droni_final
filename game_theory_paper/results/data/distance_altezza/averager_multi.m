close all
clear all
load("./results_multi_opt_dist0.mat")
load("./results_multi_opt_dist50.mat")
load("./results_multi_opt_dist150.mat")
load("./results_multi_opt_dist100.mat")

average_res_naive = results_dist_naive_0 +results_dist_naive_50+results_dist_naive_100+results_dist_naive_150;
average_res_naive=average_res_naive/4;

average_res_opt_multi = results_dist_opt_0 +results_dist_opt_50+results_dist_opt_100+results_dist_opt_150; %mod
average_res_opt_multi = average_res_opt_multi/4;

average_difference = (results_risparmio_0+results_risparmio_50+results_risparmio_100+results_risparmio_150)/4;
average_valgame =  (values_game_0+values_game_100+values_game_150)/4;
altezza = [5,10,15,30,50,70,100];

save("average_multi.mat","average_res_opt_multi","average_valgame","altezza","average_res_naive")