

% MATLAB code to anaylse results in the Advanced Research Method's 
% group D project 'Does human prototypicality ratings correlate
% with neural network categorization?'.

% Organising the data per category and a few other data manipulations

clear all

% READING THE DATA AND PREPARATIONS

%readData.m;

load('human_data.mat');
load('neuralnet_data.mat');

np=size(neuralnet_data,1); % number of pictures
nh=size(human_data,2); % number of human participants

org.category={'fruit','church','dog','house','teapot','table','airplane','coffeemug','volcano','castle','car'};
nc = size(org.category,2); % number of categories
for ic=1:nc
    load(strcat('json/',cell2mat(org.category(ic)),'.mat'));
end

% --- preparations for data organisation per category

picture=neuralnet_data.Properties.RowNames;
picture_category=neuralnet_data.Category_shuffled; % _shuffeld iso _chosen because of coffeemug
org.pc=zeros(1,nc); % number of pictures for each category
np_nc=ceil(np/nc); % pictures per category, if this is the same for each category
org.pic_ic_ip=zeros(nc, np_nc); % list of picture numbers for each category 
for ic=1:nc % for each category
    ic_ip=0;
    for ip=1:np % and for very picture
        if isempty(setdiff(picture_category(ip),org.category(ic))) % if picture category is category
            ic_ip=ic_ip+1;
            org.pic_ic_ip(ic, ic_ip)=ip; % store picture number in list
        end
        org.pc(ic)=ic_ip;
    end
end

% GETTING THE DATA IN SHAPE

% --- Human scores
human.subjects=human_data.Properties.VariableNames;
human.picture=human_data(1:np,:).Properties.RowNames;
human.score=cell2mat(table2array(human_data(1:np,1:nh)));
human.score_per_cat=zeros(nc, np_nc, nh);
for ic=1:nc
    for ic_ip=1:org.pc(ic)
        human.score_per_cat(ic,ic_ip,:)=human.score(org.pic_ic_ip(ic,ic_ip),:);
    end
end

human.mean_score=median(human.score_per_cat,3);
human.std_score=std(human.score_per_cat,0,3);
human.min_score=human.mean_score-human.std_score;
human.max_score=human.mean_score+human.std_score;

% --- Neural net probabilities
neuralnet.picture=neuralnet_data(1:110,:).Properties.RowNames;
neuralnet.piccat=neuralnet.picture;
neuralnet.picnumber=neuralnet.picture;
neuralnet.image=cell2table(neuralnet_data.Image);
neuralnet.class=cell2table(neuralnet_data.Category_shuffled);
neuralnet.score=neuralnet_data.Prob_chosen; % probability from sources 1. and 2.
neuralnet.score_per_cat=zeros(nc, np_nc);
neuralnet.json_probs=zeros(nc,np_nc,1000);
for ic=1:nc
    for ic_ip=1:org.pc(ic)
        neuralnet.score_per_cat(ic,ic_ip)=neuralnet.score(org.pic_ic_ip(ic,ic_ip),1);
        %neuralnet.score_per_cat(ic,ic_ip)=str2num(table2array(neuralnet.score{org.pic_ic_ip(ic,ic_ip),1}));
        neuralnet.picnumber=strsplit(neuralnet.picture{org.pic_ic_ip(ic,ic_ip)},'_');
        p1=str2num(cell2mat(neuralnet.picnumber(1)));
    end
end
neuralnet.score_per_cat_on_human_scale=neuralnet.score_per_cat*7;

% find the best scoring index:
max(sum(neuralnet.json_probs(1,:,:),2));
find(sum(neuralnet.json_probs(1,:,:),2)>2); % this gives 952 for fruit

human.class=neuralnet.class;

% Regression probabilities
%table=readtable('RegressionScores.txt');
%regression_subject=table2array(table(:,1));
%regression_class=table2array(table(:,2));
%regression_picture=table2array(table(:,3));
%regression_score=table2array(table(:,4));

save('human.mat','human');
save('neuralnet.mat','neuralnet');
save('org.mat','org');

