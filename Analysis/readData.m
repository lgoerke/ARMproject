
% MATLAB code to anaylse results in the Advanced Research Method's 
% group D project 'Does human prototypicality ratings correlate
% with neural network categorization?'.

% Reading the data from the survey adn neuralnet results

clear all

load('networkClassification.mat');

shuffled1;
shuffled2;
shuffled3;

query1_tbl=cell2table(query1);
query2_tbl=cell2table(query2);
query3_tbl=cell2table(query3);

% There are three sources for the neural net probabilies:
% 1. the files 'n_prob_list1.mat' etc
% 2. the file 'chosen_probs.txt'
% 3. the files 'shuffled1.txt' etc.
% The first two sources have the same lists fo probabilities, 
% but the third source is different. Why is this?
% I am guessing that the first two sources are more reliable?

prob1_tbl=cell2table([prob1,classifications]);

prob1_tbl.Properties.RowNames=query1;

p1=sortrows(prob1_tbl,'RowNames');

p1.Properties.VariableNames = {'Picture_shuffled','Category_shuffled','Prob_shuffled','Picture_chosen','Category_chosen','Prob_chosen'};

p1.Image=strcat('pictures1/',p1.Picture_shuffled,'_',p1.Category_shuffled,'.jpg');

% p1, p2, and p3 should be the same if data is correct

chosen_probs=readtable('chosen_probs.txt'); % just to check, they are the same values as in 'n_prob_list1.mat' etc.

surv1=readtable('Dataset1_Final.xlsx');
surv2=readtable('Dataset2_Final.xlsx');
surv3=readtable('Dataset3_Final.xlsx');

s1=cell2table(table2cell(surv1).');
s2=cell2table(table2cell(surv2).');
s3=cell2table(table2cell(surv3).');

Rows1={'Time','Age','Gender','Country','Cblind','Test1','Test2','Understood','Exclude',query1,'Comment','Agree'};
Rows2={'Time','Age','Gender','Country','Cblind','Test1','Test2','Understood','Exclude',query2,'Comment','Agree'};
Rows3={'Time','Age','Gender','Country','Cblind','Test1','Test2','Understood','Exclude',query3,'Comment','Agree'};

Rows1flat=[Rows1{:}];
Rows2flat=[Rows2{:}];
Rows3flat=[Rows3{:}];

s1.Properties.RowNames = [Rows1{:}];
s2.Properties.RowNames = [Rows2{:}];
s3.Properties.RowNames = [Rows3{:}];

s1.Properties.VariableNames = strcat('g1',s1.Properties.VariableNames);
s2.Properties.VariableNames = strcat('g2',s2.Properties.VariableNames);
s3.Properties.VariableNames = strcat('g3',s3.Properties.VariableNames);

d1=sortrows(s1,'RowNames');
d2=sortrows(s2,'RowNames');
d3=sortrows(s3,'RowNames');

human_data=[d1,d2,d3];
neuralnet_data=p1; 

save('human_data.mat','human_data');
save('neuralnet_data.mat','neuralnet_data');

