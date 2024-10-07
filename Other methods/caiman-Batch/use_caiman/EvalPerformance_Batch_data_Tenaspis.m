clear
% Path to where "GetPerformance_Jaccard.m" is located
addpath(genpath(''))
addpath(genpath('C:\Matlab Files\STNeuroNet-master\Software'))
% Run over Layer 275 data
ThJaccard = 0.5;
expID={'Mouse_1K','Mouse_2K','Mouse_3K','Mouse_4K',...
    'Mouse_1M','Mouse_2M','Mouse_3M','Mouse_4M'};
% gtDir = 'C:\Matlab Files\STNeuroNet-master\Markings\ABO\Layer275\FinalGT\';
% gtDir = 'D:\ABO\20 percent 200\GT Masks\';
datadir = 'D:/data_TENASPIS/added_refined_masks/';
gtDir = [datadir, 'GT Masks\'];

root = [datadir, 'caiman-Batch\']; % bin 5
if ~exist([root,'Masks\'])
    mkdir([root,'Masks\'])
end

FinalRecall = zeros(1,8); FinalPrecision = FinalRecall; FinalF1 = FinalRecall;
stat_ProcessTime = [];
for k= 1:8
    load([root,'275\',expID{k},'.mat']); %
    % Save execution time info
    stat_ProcessTime = [stat_ProcessTime;ProcessTime];
    
    % get masks and final performance
    L1=sqrt(size(Ab,1));
    [finalSegments] = ProcessOnACIDMasks(Ab,[L1,L1],0.2);  
    [FinalRecall(k), FinalPrecision(k), FinalF1(k)] = GetPerformance_Jaccard(...
        gtDir,expID{k},finalSegments,ThJaccard);
    
    %save results
    save([root,'Masks\',expID{k},'_neurons.mat'],'finalSegments','-v7.3')
end

ProcessTime = stat_ProcessTime;
% save results
% Table_time = ProcessTime;
% save([root,'Table_time.mat'],'Table_time')
Table = [FinalRecall; FinalPrecision; FinalF1; ProcessTime']';
Table_ext = [Table; mean(Table,1); std(Table,1,1)];
row=mean(Table,1)
row=reshape([mean(Table,1); std(Table,1,1)],1,[]);
save([root,'Performance_275.mat'],'Table_ext',...
    'FinalRecall','FinalPrecision','FinalF1','expID','ProcessTime','-v7.3')
