close all
clear all
load("./results_multi_opt_dist_variance0.mat")
load("./results_multi_opt_dist_variance50.mat")
load("./results_multi_opt_dist_variance100.mat")
load("./results_multi_opt_dist_variance150.mat")
load("./results_single_opt_dist_variance0.mat")
load("./results_single_opt_dist_variance50.mat")
load("./results_single_opt_dist_variance100.mat")
load("./results_single_opt_dist_variance150.mat")

%Mean Naive
mean_naive = results_dist_naive_0 +results_dist_naive_50+results_dist_naive_100+results_dist_naive_150;
mean_naive = mean_naive + results_dist_naive_multi0 +results_dist_naive_multi50+results_dist_naive_multi100+results_dist_naive_multi150;
mean_naive = mean_naive/8;
%Variance Naive
var_naive = results_variance_naive_0+results_variance_naive_50+results_variance_naive_100+results_variance_naive_150;
var_naive = var_naive+results_variance_naive_multi0+results_variance_naive_multi50+results_variance_naive_multi100+results_variance_naive_multi150;
var_naive = var_naive/8;

 %Mean Single opt
average_opt_single = results_dist_opt_0 +results_dist_opt_100+results_dist_opt_100+results_dist_opt_150; %mod
mean_one_opt = average_opt_single/4;
%Variance Single opt
var_single = results_variance_opt_0+results_variance_opt_50+results_variance_opt_100+results_variance_opt_150;
var_one_opt = var_single/4;

%Mean Multi
mean_multi = results_dist_opt_multi0 +results_dist_opt_multi100+results_dist_opt_multi50+results_dist_opt_multi150;
mean_multi = mean_multi/4;
%Variance Multi
var_multi = results_variance_opt_multi0 +results_variance_opt_multi50+results_variance_opt_multi100+results_variance_opt_multi150;
var_multi = var_multi/4;

save("processed_data.mat","mean_naive","var_naive","mean_one_opt","var_one_opt","mean_multi","var_multi")