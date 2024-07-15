clear;
addpath(genpath('../ANE'))
dir_SUNS_sub = fullfile('SUNS_FISSA_SF25','4816[1]th3');
%%
dir_video='E:\1photon-small\added_refined_masks\';
list_Exp_ID = { 'c25_59_228','c27_12_326','c28_83_210',...
                'c25_163_267','c27_114_176','c28_161_149',...
                'c25_123_348','c27_122_121','c28_163_244'};
% list_Exp_ID = list_Exp_ID(1:5);
num_Exp=length(list_Exp_ID);

dir_SUNS = fullfile(dir_video, dir_SUNS_sub);
dir_GT_masks = fullfile(dir_video,'GT Masks'); % FinalMasks_

method = 'SUNS2+TUnCaT'; % 'SUNS2-ANE+TUnCaT' % 
if contains(method,'ANE')
    dir_traces = fullfile(dir_SUNS, 'output_masks/add_new_blockwise/trained dropout 0.8exp(-15)/TUnCaT_time');
else
    dir_traces = fullfile(dir_SUNS, 'output_masks/TUnCaT_time');
end
% dir_traces = [dir_traces,'_1e-3'];
list_unmix_time = zeros(num_Exp,1);

%%
for ii = 1:num_Exp
    Exp_ID = list_Exp_ID{ii};
    load(fullfile(dir_traces,['Output_Masks_',Exp_ID,'.mat']),'Masks','traces','TUnCaT_time');
    list_unmix_time(ii) = TUnCaT_time;
end
