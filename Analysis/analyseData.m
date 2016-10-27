

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

np=size(neuralnet_data,1); % number of pictures
nh=size(human_data,2); % number of human participants

category={'fruit','church','dog','house','teapot','table','airplane','coffeemug','volcano','castle','car'};
nc = size(category,2); % number of categories

picture=neuralnet_data.Properties.RowNames;
picture_category=neuralnet_data.Category_shuffled; % _shuffeld iso _chosen because of coffeemug
pc=zeros(1,nc); % number of pictures for each category
np_nc=ceil(np/nc); % pictures per category, if this is the same for each category
pic_ic_ip=zeros(nc, np_nc); % list of picture numbers for each category 
for ic=1:nc % for each category
    ic_ip=0;
    for ip=1:np % and for very picture
        if isempty(setdiff(picture_category(ip),category(ic))) % if picture category is category
            ic_ip=ic_ip+1;
            pic_ic_ip(ic, ic_ip)=ip; % store picture number in list
        end
        pc(ic)=ic_ip;
    end
end

% GETTING THE DATA IN SHAPE

% Subject,Class,Picture,Score

% Human scores
human.subjects=human_data.Properties.VariableNames;
human.picture=human_data(1:np,:).Properties.RowNames;
human.score=cell2mat(table2array(human_data(1:np,1:nh)));
human.score_per_cat=zeros(nc, np_nc, nh);
for ic=1:nc
    for ic_ip=1:pc(ic)
        human.score_per_cat(ic,ic_ip,:)=human.score(pic_ic_ip(ic,ic_ip),:);
    end
end

human.mean_score=mean(human.score_per_cat,3);
human.min_score=min(human.score_per_cat,3);
human.max_score=max(human.score_per_cat,3);

% Neural net probabilities
neuralnet.picture=neuralnet_data(1:110,:).Properties.RowNames;
neuralnet.class=table2array(neuralnet_data(:,2));
neuralnet.score=cell2table(neuralnet_data.Prob_chosen); % probability from sources 1. and 2.
neuralnet.score_per_cat=zeros(nc, np_nc);
for ic=1:nc
    for ic_ip=1:pc(ic)
        neuralnet.score_per_cat(ic,ic_ip)=str2num(table2array(neuralnet.score{pic_ic_ip(ic,ic_ip),1}));
    end
end
neuralnet.score_per_cat_on_human_scale=neuralnet.score_per_cat*7;

human.class=neuralnet.class;

% Regression probabilities
%table=readtable('RegressionScores.txt');
%regression_subject=table2array(table(:,1));
%regression_class=table2array(table(:,2));
%regression_picture=table2array(table(:,3));
%regression_score=table2array(table(:,4));

for ic=1:nc
    figure;
    for ic_ip=1:pc(ic)
        ax=subplot(2,np_nc/2,ic_ip);
        histogram(ax,human.score_per_cat(ic,ic_ip,:));
        hold on;
        histogram(ax,neuralnet.score_per_cat_on_human_scale(ic,ic_ip,:));
        title(ax,strrep(human.picture(pic_ic_ip(ic,ic_ip)),'_',' '));
        axis(ax,[0 8 0 Inf]);
    end
end

rho_nh=zeros(nc,1); 
pval_nh=zeros(nc,1);

for ic=1:nc
    
    sh = human.mean_score(ic,:).'; % human scores
    sn = neuralnet.score_per_cat(ic,:).'; % human scores
    %sr = regression_score(ic,pc(ic)); % human scores
    wh = ones(pc(ic),1);
    %wh = [1;1;1;1];
    wr = [1;1;1;1];

    % CORRELATION FITTED TO MONOTONOUS FUNCTION

    % Fit of human scores versus neural net probabilities
    mdlFun_nh = @(b,x) b(1) + b(2)*x;
    %mdlFun = @(b,x) b(1).*(1-exp(-b(2).*x));
    start = [0, 0];
    mdl_nh = fitnlm(sn,sh,mdlFun_nh,start,'Weight',wh);
    res_nh = sh - predict(mdl_nh,sort(sn));

    % Fit of regression probabilities versus neural net probabilities
    %mdlFun_nr = @(b,x) b(1) + b(2)*x;
    %start = [0, 0]
    %mdl_nr = fitnlm(sn,sr,mdlFun_nr,start,'Weight',wr);
    %res_nr = sr - predict(mdl_nr,sort(sn));

    % SPEARMAN'S RANK CORRELATION

    % The article by Lake et al uses Spearman correlation
    [rho_nh(ic), pval_nh(ic)] = corr(sn, sh);
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
    plot(ax3,sort(sn),zeros(1,np_nc),'r-');

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
    
end

figure;
bar(rho_nh); % wanted to add category labels but bar does not allow
bar(pval_nh); % wanted to add category labels but bar does not allow
