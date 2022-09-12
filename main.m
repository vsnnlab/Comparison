clear; clc;
close all;

%% Notes
% Demo codes for
% "Comparison of visual quantities in untrained deep neural networks"
% Hyeonsu Lee, Woochul Choi, Dongil Lee, and Se-Bum Paik*

% *Contact: sbpaik@kaist.ac.kr

% This code takes 4-6 minutes for running. (May depend on your computer setting) 

% Prerequirement 
% 1) MATLAB 2021a or later version
% 2) Installation of the Deep Learning Toolbox
%    (https://www.mathworks.com/products/deep-learning.html)
% 3) Installation of the pretrained AlexNet
%    (https://de.mathworks.com/matlabcentral/fileexchange/59133-deep-learning-toolbox-model-for-alexnet-network)
% 4) Download stimulus sets and pre-trained alexnet
%    (https://drive.google.com/drive/folders/1fBqvA3EdyJZn0we5lohzAh4S9eWJV7f-?usp=sharing)

%% code
toolbox_chk;
warning('off');

tic;

addpath(genpath(pwd)); % set path

seed = 4; % random seed

load('alexnet_pretrained_2021a.mat');

layerList_target = {'relu1','relu2','relu3','relu4','relu5'};
num_unit = [290400 186624 64896 64896 43264];

list_xlabel = {{'0/6','1/6','2/6','3/6','4/6','5/6','6/6'},...
    {'-6','-4','-2','0','+2','+4','+6'}};

colorMat = [126 47 142; 235 97 1]/255;

%% main
rng(seed);

idxShuffle = [2 6 10 12 14]; % index for convolution layer
net_test = randomizeNet(net, idxShuffle); % random initialized network

%% R1) Find selective units
getUnit_L5;

%% R2) SVM
type_i = 1; % 1: Proportion, 2: Difference comparison
svmComparison;

%% R3) Connectivity: Inc/Dec > Prop/Diff
getIncDecUnit;
getUnit_L3;

type_i = 1; % 1: Proportion, 2: Difference units
checkConnectivity;

%% R4) Nonlinearity of IncDec responses
checkNonlinearity;

toc;





