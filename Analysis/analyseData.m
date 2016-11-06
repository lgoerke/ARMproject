

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
load('logreg.mat');
load('org.mat');

nc=size(neuralnet.score_per_cat,1);
np=size(neuralnet.picture,1);
np_nc=ceil(np/nc); % pictures per category, if this is the same for each category


% PLOTTING THE DATA PER PICTURE FOR ANALYSIS

% ttests?
%h = ttest(human.score_per_cat(1,1,:),neuralnet.score_per_cat_on_human_scale(1,1));

% --- Histogram plots of human and neural net data with pictures

ifig=0;
for ic=1:nc
    %figure('units','normalized','outerposition',[0 0 1 1]);
    for ic_ip=1:np_nc
        %ax1=subplot(4,np_nc/2,ic_ip+10/2*floor(2*(ic_ip-1)/10));
        %ax1=subplot(4,np_nc/2,ic_ip+org.pc(ic)/2*floor(2*(ic_ip-1)/org.pc(ic)));
        %histogram(ax1,human.score_per_cat(ic,ic_ip,:));
        %hold on;
        %nn=ones(10,1);
        %nn=nn*neuralnet.score_per_cat_on_human_scale(ic,ic_ip,:);
        %histogram(ax1,nn,'FaceColor','k');
        %title(ax1,strrep(human.picture(org.pic_ic_ip(ic,ic_ip)),'_',' '));
        %axis(ax1,[0 8 0 50]);
        %ax2=subplot(4,np_nc/2,ic_ip+org.pc(ic)/2*(floor(2*(ic_ip-1)/org.pc(ic))+1));
        %nn_image=cell2mat(neuralnet.image{org.pic_ic_ip(ic,ic_ip),1});
        %imagesc(ax2,imread(strrep(nn_image,'coffeemug','coffee mug')));
        %axis(ax2,'off');
    end
    %ifig=ifig+1;
    %pdffile=cell2mat(strcat('pdf/',org.category(ic),'_hist.pdf'));
    %save2pdf(pdffile,ifig,500);
end

% STATISTICAL ANALYSIS

rho_nh=zeros(nc,1); 
pval_nh=zeros(nc,1);
rho_rh=zeros(nc,1); 
pval_rh=zeros(nc,1);

sh=zeros(np_nc,nc);
shs=zeros(np_nc,nc);
shl=zeros(np_nc,nc);
shh=zeros(np_nc,nc);
sn=zeros(np_nc,nc);
sr=zeros(np_nc,nc);
sfn=zeros(np_nc,nc);

for ic=1:nc
    
    sh(:,ic) = human.mean_score(ic,:).'; % human scores
    shs(:,ic)= human.std_score(ic,:).'; % human scores
    shl(:,ic) = human.min_score(ic,:).'; % human scores
    shh(:,ic) = human.max_score(ic,:).'; % human scores
    sn(:,ic) = neuralnet.score_per_cat(ic,:).'; % neural net scores
    sr(:,ic) = logreg.score_per_cat(ic,:); % regression scores
    wh = 1./(0.5*human.std_score(ic,:).'); %ones(org.pc(ic),1);
    %wh = [1;1;1;1];
    wr = [1;1;1;1];
    
    % --- correlation fitted to monotonous function with weights 1/std

    % Fit of human scores versus neural net probabilities
    mdlFun = @(b,x) b(1) + b(2)*x;
    start = [0, 0];
    mdl_nh = fitnlm(sn(:,ic),sh(:,ic),mdlFun,start,'Weight',wh);
    
    sfn(:,ic) = predict(mdl_nh,sort(sn(:,ic)));

    % Fit of regression probabilities versus neural net probabilities
    mdlFun = @(b,x) b(1) + b(2)*x;
    start = [0, 0];
    mdl_rh = fitnlm(sr(:,ic),sh(:,ic),mdlFun,start,'Weight',wh);
    
    sfr(:,ic) = predict(mdl_rh,sort(sr(:,ic)));

    % --- Spearman's rank correlation

    % The article by Lake et al uses Spearman correlation
    [rho_nh(ic), pval_nh(ic)] = corr(sn(:,ic), sh(:,ic));
    [rho_rh(ic), pval_rh(ic)] = corr(sr(:,ic), sh(:,ic));

    % --- regression plots

    %figure;%('units','normalized','outerposition',[0 0 1 1]);
    %ax1 = subplot(1,2,1);
    %%plot(ax1, sn, sh, 'bo');
    %errorbar(ax1, sn(:,ic), sh(:,ic), shs(:,ic),'bo');
    %hold on;
    %plot(ax1,sort(sn(:,ic)),sfn(:,ic),'r-');

    %ax2 = subplot(1,2,2);
    %%plot(ax2, sn, sr, 'bo');
    %errorbar(ax2, sr(:,ic), sh(:,ic), shs(:,ic),'bo');
    %hold on;
    %plot(ax2,sort(sr(:,ic)),sfr(:,ic),'r-');

    % --- plotting layout
    %title(ax1,strcat(org.category(ic),': human - neural net comparison'));
    %xlim(ax1, [0 1.1]);
    %ylim(ax1, [0 8]);
    %xlabel(ax1, 'neural net probability');
    %ylabel(ax1, 'human score');
    %title(ax2,strcat(org.category(ic),': regression - neural net comparison'));
    %xlim(ax2, [0 1.1]);
    %ylim(ax2, [0 1.1]);
    %xlabel(ax2, 'neural net probability');
    %ylabel(ax2, 'regression probability');

    ifig=ifig+1;
    pdffile=cell2mat(strcat('pdf/',org.category(ic),'_regr.pdf'));
    %save2pdf(strcat(pdffile),ifig,500);
    
end

figure;%('units','normalized','outerposition',[0 0 1 1]);
ax1=subplot(1,2,1);
bar(ax1,rho_nh );% wanted to add category labels but bar does not allow
set(gca,'XTickLabel',org.category);
ax1.XTickLabelRotation=90;
title(ax1,'Pearson pairwise linear correlation coefficient ');
ax2=subplot(1,2,2);
bar(ax2,pval_nh); % wanted to add category labels but bar does not allow
set(gca,'XTickLabel',org.category);
ax2.XTickLabelRotation=90;
title(ax2,'Pearson p value');

ifig=ifig+1;
%save2pdf('pdf/spearman.pdf',ifig,500);


figure('units','normalized','outerposition',[0 0 1 1]);
orient landscape;

ax1 = subplot(2,2,1);
errorbar(ax1, sn(:,1), sh(:,1), shs(:,1),'bo');
hold on;
plot(ax1,sort(sn(:,1)),sfn(:,1),'r-');
title(ax1,'fruit');
xlim(ax1, [0 1.1]);
ylim(ax1, [0 8]);

ax2 = subplot(2,2,2);
errorbar(ax2, sn(:,3), sh(:,3), shs(:,3),'bo');
hold on;
plot(ax2,sort(sn(:,3)),sfn(:,3),'r-');
title(ax2,'dog');
xlim(ax2, [0 1.1]);
ylim(ax2, [0 8]);

ax3 = subplot(2,2,3);
errorbar(ax3, sn(:,7), sh(:,7), shs(:,7),'bo');
hold on;
plot(ax3,sort(sn(:,7)),sfn(:,7),'r-');
title(ax3,'airplane');
xlim(ax3, [0 1.1]);
ylim(ax3, [0 8]);

ax4 = subplot(2,2,4);
errorbar(ax4, sn(:,9), sh(:,9), shs(:,9),'bo');
hold on;
plot(ax4,sort(sn(:,9)),sfn(:,9),'r-');
title(ax4,'volcano');
xlim(ax4, [0 1.1]);
ylim(ax4, [0 8]);

%save2pdf('pdf/fourplotsforsignificantcategories.pdf',ifig,500);

figure('units','normalized','outerposition',[0 0 1 1]);
orient landscape;

ax1 = subplot(2,2,1);
errorbar(ax1, sr(:,1), sh(:,1), shs(:,1),'bo');
hold on;
plot(ax1,sort(sr(:,1)),sfr(:,1),'r-');
title(ax1,'fruit');
xlim(ax1, [0 1.1]);
ylim(ax1, [0 8]);

ax2 = subplot(2,2,2);
errorbar(ax2, sr(:,3), sh(:,3), shs(:,3),'bo');
hold on;
plot(ax2,sort(sr(:,3)),sfr(:,3),'r-');
title(ax2,'dog');
xlim(ax2, [0 1.1]);
ylim(ax2, [0 8]);

ax3 = subplot(2,2,3);
errorbar(ax3, sr(:,7), sh(:,7), shs(:,7),'bo');
hold on;
plot(ax3,sort(sr(:,7)),sfr(:,7),'r-');
title(ax3,'airplane');
xlim(ax3, [0 1.1]);
ylim(ax3, [0 8]);

ax4 = subplot(2,2,4);
errorbar(ax4, sr(:,9), sh(:,9), shs(:,9),'bo');
hold on;
plot(ax4,sort(sr(:,9)),sfr(:,9),'r-');
title(ax4,'volcano');
xlim(ax4, [0 1.1]);
ylim(ax4, [0 8]);

