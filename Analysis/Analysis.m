

% MATLAB code to anaylse results in the Advanced Research Method's 
% group D project 'Does human prototypicality ratings correlate
% with neural network categorization?'.

% The neural net scores are correlated to the human scores, 
% as well as the baseline regression scores, by a yet to be
% specified monotonous function. This may be simply a linear fit,
% but because there is no reason a-priori why the two scales 
% should be related linearly, the option is kept open to use 
% another relationship. A weighting can be applied to each data
% point (= each image), possibly depending on the spread in the 
% scores.

% The article by Lake et al does not use a linear fit, for the
% reason mentioned above, but Spearman's rank correlation (rho).
% This is also calculated here.

clear all

% READING THE DATA

%readData.m;

load('human_data.mat');
load('neuralnet_data.mat');

np = size(neuralnet_data,1);

classes=['fruit','church','dog','house','teapot','table','airplane','coffee_mug','volcano','castle'];
nc = size(classes,2)

pictures=neuralnet_data.Properties.RowNames;
picture_classes=neuralnet_data.Class_shuffled;

% GETTING THE DATA IN SHAPE

% Subject,Class,Picture,Score

% Human scores
HumanSubjects=human_data.Properties.VariableNames;
HumanPicture=human_data(1:np,:).Properties.RowNames;
HumanScores=cell2mat(table2array(human_data(1:np,1:74)));
HumanMeanScore=mean(HumanScores,2);
HumanMinScore=min(HumanScores,2);
HumanMaxScore=max(HumanScores,2);

% Neural net probabilities
NeuralNetPicture=neuralnet_data(1:110,:).Properties.RowNames;
NeuralNetClass=table2array(neuralnet_data(:,2));
NeuralNetScore=neuralnet_data.Prob_chosen; % probability from sources 1. and 2.
NeuralNetSc=NeuralNetScore;
for ip=1:np
    NeuralNetSc{ip}=str2num(NeuralNetSc{ip,1});
end
NeuralNetScore=cell2mat(NeuralNetSc);

HumanClass=NeuralNetClass;

% Regression probabilities
%table=readtable('RegressionScores.txt');
%RegressionSubject=table2array(table(:,1));
%RegressionClass=table2array(table(:,2));
%RegressionPicture=table2array(table(:,3));
%RegressionScore=table2array(table(:,4));



sh = HumanMeanScore(1:np); % human scores
sn = NeuralNetScore(1:np); % human scores
%sr = RegressionScore(1:np); % human scores
wh = ones(np,1);
%wh = [1;1;1;1];
wr = [1;1;1;1];

% CORRELATION FITTED TO MONOTONOUS FUNCTION

% Fit of human scores versus neural net probabilities
mdlFun_nh = @(b,x) b(1) + b(2)*x;
%mdlFun = @(b,x) b(1).*(1-exp(-b(2).*x));
start = [0, 0]
mdl_nh = fitnlm(sn,sh,mdlFun_nh,start,'Weight',wh);
res_nh = sh - predict(mdl_nh,sort(sn));

% Fit of regression probabilities versus neural net probabilities
%mdlFun_nr = @(b,x) b(1) + b(2)*x;
%start = [0, 0]
%mdl_nr = fitnlm(sn,sr,mdlFun_nr,start,'Weight',wr);
%res_nr = sr - predict(mdl_nr,sort(sn));

% SPEARMAN'S RANK CORRELATION

% The article by Lake et al uses Spearman correlation
[rho_nh, pval_nh] = corr(sn, sh);
%[rho_nr, pval_nr] = corr(sn, sr);

% PLOTTING DETAILS

figure;
ax1 = subplot(2,2,1);
plot(ax1, sn, sh, 'bo');
hold on;
plot(ax1,sort(sn),predict(mdl_nh,sort(sn)),'r-');

ax2 = subplot(2,2,2);
%plot(ax2, sn, sr, 'bo');
%old on;
%plot(ax2,sort(sn),predict(mdl_nr,sort(sn)),'r-');

ax3 = subplot(2,2,3);
plot(ax3, sn, res_nh, 'bo');
hold on;
plot(ax3,sort(sn),zeros(1,np),'r-');

ax4 = subplot(2,2,4);
%plot(ax4, sn, res_nr, 'bo');
hold on;
%plot(ax4,sort(sn),zeros(1,np),'r-');

% plot layout
title(ax1,'human score versus neural net probability');
xlim(ax1, [0 1.1]);
ylim(ax1, [0 8]);
xlabel(ax1, 'neural net probability');
ylabel(ax1, 'human score');
title(ax2,'regression versus neural net probability');
xlim(ax2, [0 1.1]);
ylim(ax2, [0 1.1]);
xlabel(ax2, 'neural net probability');
ylabel(ax2, 'regression probability');
title(ax3,'residuals for fit of human scores');
xlim(ax3, [0 1.1]);
ylim(ax3, [-8 8]);
xlabel(ax3, 'neural net probability');
ylabel(ax3, 'human score');
title(ax4,'residuals for fit of regression probabilities');
xlim(ax4, [0 1.1]);
ylim(ax4, [-1.1 1.1]);
xlabel(ax4, 'neural net probability');
ylabel(ax4, 'regression probability');

%coeffvalues(fit_nh)
%confint_nh = confint(fit_nh)
%fit_nr = fit(sn,sr,'poly2')
%plot(fit_nh,sn,sh,'Residuals')
%[fit_nh,gof,output] = fit(sn,sh,'poly2')%,'normalize','on')

