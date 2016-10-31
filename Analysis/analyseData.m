

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

load('human.mat');
load('neuralnet.mat');
load('org.mat');

nc=size(neuralnet.score_per_cat,1);
np=size(neuralnet.picture,1);
np_nc=ceil(np/nc); % pictures per category, if this is the same for each category


% PLOTTING THE DATA PER PICTURE FOR ANALYSIS

% ttests?
h = ttest(human.score_per_cat(1,1,:),neuralnet.score_per_cat_on_human_scale(1,1));

% --- Histogram plots of human and neural net data with pictures

ifig=0;
for ic=1:nc
    figure;%('units','normalized','outerposition',[0 0 1 1]);
    for ic_ip=1:np_nc
        ax1=subplot(4,np_nc/2,ic_ip+org.pc(ic)/2*floor(2*(ic_ip-1)/org.pc(ic)));
        histogram(ax1,human.score_per_cat(ic,ic_ip,:));
        hold on;
        histogram(ax1,neuralnet.score_per_cat_on_human_scale(ic,ic_ip,:),'LineWidth',2,'FaceColor','k');
        title(ax1,strrep(human.picture(org.pic_ic_ip(ic,ic_ip)),'_',' '));
        axis(ax1,[0 8 0 50]);
        ax2=subplot(4,np_nc/2,ic_ip+org.pc(ic)/2*(floor(2*(ic_ip-1)/org.pc(ic))+1));
        nn_image=cell2mat(neuralnet.image{org.pic_ic_ip(ic,ic_ip),1});
        imagesc(ax2,imread(strrep(nn_image,'coffeemug','coffee mug')));
        axis(ax2,'off');
    end
    ifig=ifig+1;
    pdffile=cell2mat(strcat('pdf/',org.category(ic),'_hist.pdf'));
    save2pdf(pdffile,ifig,500);
end

% STATISTICAL ANALYSIS

rho_nh=zeros(nc,1); 
pval_nh=zeros(nc,1);

for ic=1:nc
    
    sh = human.mean_score(ic,:).'; % human scores
    shs= human.std_score(ic,:).'; % human scores
    shl = human.min_score(ic,:).'; % human scores
    shh = human.max_score(ic,:).'; % human scores
    sn = neuralnet.score_per_cat(ic,:).'; % neural net scores
    %sr = regression_score(ic,org.pc(ic)); % regression scores
    wh = 1./(0.5*human.std_score(ic,:).'); %ones(org.pc(ic),1);
    %wh = [1;1;1;1];
    wr = [1;1;1;1];
    
    % --- correlation fitted to monotonous function with weights 1/std

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

    % --- Spearman's rank correlation

    % The article by Lake et al uses Spearman correlation
    [rho_nh(ic), pval_nh(ic)] = corr(sn, sh);
    %[rho_nr, pval_nr] = corr(sn, sr);

    % --- regression plots

    figure;%('units','normalized','outerposition',[0 0 1 1]);
    ax1 = subplot(2,2,1);
    %plot(ax1, sn, sh, 'bo');
    errorbar(ax1, sn, sh, shs,'bo');
    hold on;
    plot(ax1,sort(sn),predict(mdl_nh,sort(sn)),'r-');

    ax2 = subplot(2,2,2);
    %plot(ax2, sn, sr, 'bo');
    %old on;
    %plot(ax2,sort(sn),predict(mdl_nr,sort(sn)),'r-');

    ax3 = subplot(2,2,3);
    errorbar(ax3, sn, res_nh, shs, 'bo');
    hold on;
    plot(ax3,sort(sn),zeros(1,np_nc),'r-');

    ax4 = subplot(2,2,4);
    %plot(ax4, sn, res_nr, 'bo');
    hold on;
    %plot(ax4,sort(sn),zeros(1,np),'r-');

    % --- plotting layout
    title(ax1,strcat(org.category(ic),': human - neural net comparison'));
    xlim(ax1, [0 1.1]);
    ylim(ax1, [0 8]);
    xlabel(ax1, 'neural net probability');
    ylabel(ax1, 'human score');
    title(ax2,strcat(org.category(ic),': regression - neural net comparison'));
    xlim(ax2, [0 1.1]);
    ylim(ax2, [0 1.1]);
    xlabel(ax2, 'neural net probability');
    ylabel(ax2, 'regression probability');
    title(ax3,'residuals for fit of human scores');
    xlim(ax3, [0 1.1]);
    ylim(ax3, [-4 4]);
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
    
    ifig=ifig+1;
    pdffile=cell2mat(strcat('pdf/',org.category(ic),'_regr.pdf'));
    save2pdf(strcat(pdffile),ifig,500);
    
end

figure;%('units','normalized','outerposition',[0 0 1 1]);
ax1=subplot(1,2,1);
bar(ax1,rho_nh );% wanted to add category labels but bar does not allow
set(gca,'XTickLabel',org.category);
ax1.XTickLabelRotation=90;
ax2=subplot(1,2,2);
bar(ax2,pval_nh); % wanted to add category labels but bar does not allow
set(gca,'XTickLabel',org.category);
ax2.XTickLabelRotation=90;

ifig=ifig+1;
save2pdf('pdf/spearman.pdf',ifig,500);

