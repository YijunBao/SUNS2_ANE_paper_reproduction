addpath(genpath('../ANE'))
%% Set data and folder
scale_lowBG = 5e3;
scale_noise = 1;
data_name = sprintf('lowBG=%.0e,poisson=%g',scale_lowBG,scale_noise);

path_name = fullfile('../data/data_simulation',data_name);
num_Exp = 10;
list_Exp_ID = arrayfun(@(x) ['sim_',num2str(x)],0:(num_Exp-1), 'UniformOutput',false);
dir_GT_masks = fullfile(path_name,'GT Masks'); % FinalMasks_
dir_GT_info = fullfile(path_name,'GT info'); % FinalMasks_

save_date = '20230218';
dir_save = fullfile(path_name,'min1pipe');
dir_sub_save = ['cv_save_',save_date];
[list_corr_unmix, list_corr_unmix_match] = deal(cell(num_Exp, 1));
list_crop = 0:50:95;
num_crop = length(list_crop);

for cv = 1:num_Exp
    for eid = cv % trains % 1:num_Exp
        Exp_ID = list_Exp_ID{eid};
        filename = [Exp_ID,'.h5'];
        load(fullfile(dir_GT_info,['GT_',Exp_ID,'.mat']),'C');
        load(fullfile(dir_save,dir_sub_save,[Exp_ID,'_Masks.mat']),'Masks3','traces');
        ncells = size(traces,1);
        
        %% Find matched neurons
        [SpatialRecall, SpatialPrecision, SpatialF1,m] = GetPerformance_Jaccard(...
            dir_GT_masks,Exp_ID,Masks3,0.5);
        [select_seg, select_GT] = find(m');
        n_match = length(select_GT);

        %% Calculate the correlation of temporal traces
        [corr_unmix, corr_unmix_match] = deal(zeros(n_match,num_crop));
        for c = 1:num_crop
            clip_pct = prctile(traces,list_crop(c),2);
%                 clip_pct = median(traces,2);
            for n = 1:n_match
                corr_unmix(select_seg(n),c) = corr(C(select_GT(n),:)',...
                    max(clip_pct(select_seg(n)),traces(select_seg(n),:)'));
                corr_unmix_match(n,c) = corr_unmix(select_seg(n),c);
            end
        end
        corr_unmix(isnan(corr_unmix))=0;
        corr_unmix_match(isnan(corr_unmix_match))=0;
        list_corr_unmix{eid} = corr_unmix;
        list_corr_unmix_match{eid} = corr_unmix_match;
    end
end

%% Summarize
mean_corr = cell2mat(cellfun(@mean, list_corr_unmix, 'UniformOutput', false));
mean_corr_match = cell2mat(cellfun(@mean, list_corr_unmix_match, 'UniformOutput', false));
Table_mean_corr = [mean_corr; mean(mean_corr); std(mean_corr)];
Table_mean_corr_match = [mean_corr_match; mean(mean_corr_match); std(mean_corr_match)];
save(fullfile(dir_save,dir_sub_save,'mean_corr.mat'),'list_crop','mean_corr','mean_corr_match','Table_mean_corr','Table_mean_corr_match');

