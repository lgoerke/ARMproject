

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

% --- Re-calculate neural net probabilities from .json files

jsonfiles.table_i=[526, 736, 532];
jsonfiles.car_i=[407, 436, 468, 511, 609, 627, 656, 661, 751, 817];
jsonfiles.airplane_i=[404];
jsonfiles.church_i=[497];
jsonfiles.fruit_i=[988, 989, 998, 952, 953, 954, 955, 956, 957, 948, 949, 950, 951, 990, 984, 987, 948];
jsonfiles.house_i=[698, 663];
jsonfiles.dog_i=[251, 268, 256, 253, 255, 254, 257, 159, 211, 210, 212, 214, 213, 216, 215, 219, 220, 221, 217, 218, 207, 209, 206, 205, 208, 193, 202, 194, 191, 204, 187, 203, 185, 192, 183, 199, 195, 181, 184, 201, 186, 200, 182, 188, 189, 190, 197, 196, 198, 179, 180, 177, 178, 175, 163, 174, 176, 160, 162, 161, 164, 168, 173, 170, 169, 165, 166, 167, 172, 171, 264, 263, 266, 265, 267, 262, 246, 242, 243, 248, 247, 229, 233, 234, 228, 231, 232, 230, 227, 226, 235, 225, 224, 223, 222, 236, 252, 237, 250, 249, 241, 239, 238, 240, 244, 245, 259, 261, 260, 258, 154, 153, 158, 152, 155, 151, 157, 156];
jsonfiles.teapot_i=[849]; 
jsonfiles.castle_i=[483];
jsonfiles.volcano_i=[980];
jsonfiles.coffeemug_i=[504];

jsonfiles.json=zeros(500,nc);
for i=1:size(jsonfiles.fruit_i,2) 
    jsonfiles.json(:,1)=jsonfiles.json(:,1)+fruit(:,jsonfiles.fruit_i(i)+1);
end
for i=1:size(jsonfiles.church_i,2) 
    jsonfiles.json(:,2)=jsonfiles.json(:,2)+church(:,jsonfiles.church_i(i)+1);
end
for i=1:size(jsonfiles.dog_i,2) 
    jsonfiles.json(:,3)=jsonfiles.json(:,3)+dog(:,jsonfiles.dog_i(i)+1);
end
for i=1:size(jsonfiles.house_i,2)
    jsonfiles.json(:,4)=jsonfiles.json(:,4)+house(:,jsonfiles.house_i(i)+1);
end
for i=1:size(jsonfiles.teapot_i,2)
    jsonfiles.json(:,5)=jsonfiles.json(:,5)+teapot(:,jsonfiles.teapot_i(i)+1);
end
for i=1:size(jsonfiles.table_i,2)
    jsonfiles.json(:,6)=jsonfiles.json(:,6)+table(:,jsonfiles.table_i(i)+1);
end
for i=1:size(jsonfiles.airplane_i,2)
    jsonfiles.json(:,7)=jsonfiles.json(:,7)+airplane(:,jsonfiles.airplane_i(i)+1);
end
for i=1:size(jsonfiles.coffeemug_i,2)
    jsonfiles.json(:,8)=jsonfiles.json(:,8)+coffeemug(:,jsonfiles.coffeemug_i(i)+1);
end
for i=1:size(jsonfiles.volcano_i,2)
    jsonfiles.json(:,9)=jsonfiles.json(:,9)+volcano(:,jsonfiles.volcano_i(i)+1);
end
for i=1:size(jsonfiles.castle_i,2)
    jsonfiles.json(:,10)=jsonfiles.json(:,10)+castle(:,jsonfiles.castle_i(i)+1);
end
for i=1:size(jsonfiles.car_i,2)
    jsonfiles.json(:,11)=jsonfiles.json(:,11)+car(:,jsonfiles.car_i(i)+1);
end

jsonfiles.jsonall=zeros(nc,500,1000);
jsonfiles.jsonall(1,:,:)=fruit;
jsonfiles.jsonall(2,:,:)=church;
jsonfiles.jsonall(3,:,:)=dog;
jsonfiles.jsonall(4,:,:)=house;
jsonfiles.jsonall(5,:,:)=teapot;
jsonfiles.jsonall(6,:,:)=table;
jsonfiles.jsonall(7,:,:)=airplane;
jsonfiles.jsonall(8,:,:)=airplane;%coffeemug;
jsonfiles.jsonall(9,:,:)=volcano;
jsonfiles.jsonall(10,:,:)=castle;
jsonfiles.jsonall(11,:,:)=car;

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
neuralnet.score=cell2table(neuralnet_data.Prob_chosen); % probability from sources 1. and 2.
neuralnet.score_per_cat=zeros(nc, np_nc);
neuralnet.json_probs=zeros(nc,np_nc,1000);
for ic=1:nc
    for ic_ip=1:org.pc(ic)
        neuralnet.score_per_cat(ic,ic_ip)=str2num(table2array(neuralnet.score{org.pic_ic_ip(ic,ic_ip),1}));
        neuralnet.picnumber=strsplit(neuralnet.picture{org.pic_ic_ip(ic,ic_ip)},'_');
        p1=str2num(cell2mat(neuralnet.picnumber(1)));
        neuralnet.json_probs(ic,ic_ip,:)=jsonfiles.jsonall(ic,ic_ip,:);
        %
        % overwrite neural net probs with probs from 'fruit.json' etc. 
        % comment out for using probs from 'n_prob_list1.mat' etc.
        %
        neuralnet.score_per_cat(ic,ic_ip)=jsonfiles.json(p1,ic);
    end
    cat_pics(:,:)=neuralnet.json_probs(ic,:,:);
    %figure;
    %surf(cat_pics);
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

