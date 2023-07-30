clear; clc;
close all;

%% Notes
% Demo codes for
% "Comparison of visual quantities in untrained neural networks"
% Hyeonsu Lee, Woochul Choi, Dongil Lee, and Se-Bum Paik*

% *Contact: sbpaik@kaist.ac.kr

% This code takes < 5 minutes for running. (May depend on your computer setting)

% Prerequirement
% 1) MATLAB 2021a or later version
% 2) Installation of the Deep Learning Toolbox
%    (https://www.mathworks.com/products/deep-learning.html)
% 3) Installation of the pretrained AlexNet
%    (https://de.mathworks.com/matlabcentral/fileexchange/59133-deep-learning-toolbox-model-for-alexnet-network)

% Further information, or updated version: https://github.com/vsnnlab/Comparison

%% Environment check
toolbox_chk;
warning('off');

%% Result switch
switchFig1 = 1;
switchFig2 = 1;
switchFig3 = 1;
switchFig4 = 1;

%% Settings
tic;

addpath(genpath(pwd)); % set path

seed = 1; % random seed: Set for sample responses

net = alexnet;

layerList_target = {'relu1','relu2','relu3','relu4','relu5'};
num_unit = [290400 186624 64896 64896 43264];

list_xlabel = {{'0/6','1/6','2/6','3/6','4/6','5/6','6/6'},...
    {'-6','-4','-2','0','+2','+4','+6'}};

colorMat = [126 47 142; 235 97 1]/255;

%% main
rng(seed);

idxShuffle = [2 6 10 12 14]; % index for convolution layer
net_test = randomizeNet(net, idxShuffle); % random initialized network

%% Fig 1. Find proportion- and difference-selective units
if switchFig1
    disp('%%% Figure 1 running %%%');
    getUnit_L5;
    toc;
end

%% Fig 2. Proportion comparison task
if switchFig2
    disp('%%% Figure 2 running %%%');
    svmComparison;
    toc;
end

%% Fig 3. Difference comparison task (Congruent & Incongruent)
if switchFig3
    disp('%%% Figure 3 running %%%');
    svmComparison_CongIncong;
    toc;
end

%% Fig 4. Mathematical Models
if switchFig4
    disp('%%% Figure 4 running %%%');
    set_i = 1; % 1 = Proportion axis, 2 = Difference axis
    PE_i = 1; % 1 = Power function, 2 = Exponential function
    mathematicalModel;
    toc;
end
