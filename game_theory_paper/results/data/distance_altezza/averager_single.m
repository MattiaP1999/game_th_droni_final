close all
clear all
load("./results_single_opt_dist0.mat")
load("./results_single_opt_dist150.mat")
load("./results_single_opt_dist50.mat")
load("./results_single_opt_dist100.mat")

average_res_naive_single = results_dist_naive_0 +results_dist_naive_50+results_dist_naive_100+results_dist_naive_150;
average_res_naive_single=average_res_naive_single/4;

average_res_opt_single = results_dist_opt_0 +results_dist_opt_100+results_dist_opt_100+results_dist_opt_150; %mod
average_res_opt_single = average_res_opt_single/4;

average_difference = (results_risparmio_0+results_risparmio_50+results_risparmio_100+results_risparmio_150)/4;
average_valgame =  (values_game_0+values_game_100+values_game_150)/4;
altezza = [10,20,30,50,70,100];

save("average_single.mat","average_res_opt_single","average_valgame","altezza","average_res_naive_single")