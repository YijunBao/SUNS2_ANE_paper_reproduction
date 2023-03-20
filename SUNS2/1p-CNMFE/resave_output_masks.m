clear;

%% neurons and masks frame
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
list_patch_dims = [120,120; 80,80; 88,88; 192,240]; 
rate_hz = [10,15,7.5,5]; % frame rate of each video
radius = [5,6,6,6];

data_ind = 1;
data_name = list_data_names{data_ind};
dir_video = fullfile('E:\data_CNMFE',data_name);
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
num_Exp = length(list_Exp_ID);

%%
gSiz = 2*radius(data_ind); 
if data_ind == 1
    dir_sub_SUNS_TUnCaT = 'complete_TUnCaT\4816[1]th4';
    dir_sub_SUNS_FISSA = 'complete_FISSA\4816[1]th4';
%     saved_date_CNMFE = '20220521';
%     saved_date_MIN1PIPE = '20220522';
    saved_date_CNMFE = '20220605';
    saved_date_MIN1PIPE = '20220606';
    range_SNR = 0:5:100;
elseif data_ind == 2
    dir_sub_SUNS_TUnCaT = 'complete_TUnCaT\4816[1]th4';
    dir_sub_SUNS_FISSA = 'complete_FISSA\4816[1]th3';
%     saved_date_CNMFE = '20220519';
%     saved_date_MIN1PIPE = '20220519';
    saved_date_CNMFE = '20220605';
    saved_date_MIN1PIPE = '20220607';
    range_SNR = 0:2:40;
end

dir_eval_CNMFE = 'C:\Other methods\CNMF_E-1.1.2';
load(fullfile(dir_eval_CNMFE,['eval_',data_name,'_thb ',saved_date_CNMFE,' cv.mat']),'Table_time_ext');
Table_time_ext_CNMFE = Table_time_ext;
% load(fullfile(dir_eval_CNMFE,['eval_',data_name,'_thb history ',saved_date,' cv1.mat']),'list_params');

dir_eval_MIN1PIPE = 'C:\Other methods\MIN1PIPE-3.0.0';
load(fullfile(dir_eval_MIN1PIPE,['eval_',data_name,'_thb ',saved_date_MIN1PIPE,' cv.mat']),'Table_time_ext');
Table_time_ext_MIN1PIPE = Table_time_ext;
% load(fullfile(dir_eval_CNMFE,['eval_',data_name,'_thb history ',saved_date,' cv1.mat']),'list_params');
    
Lbin = length(range_SNR)-1;
NGT_PSNR = zeros(Lbin,num_Exp);
num_method = 4;
N_PSNR = zeros(Lbin,num_method,num_Exp);
Nfind_PSNR = zeros(Lbin,num_method,num_Exp);
threshJ=0.5;

%% Load video
for cv=1:num_Exp
    Exp_ID=list_Exp_ID{cv};
    for method = 3:num_method
        %% Load output masks
        switch method
            case 1 % SUNS with TUnCaT
                dir_resave = fullfile(dir_video,'complete_TUnCaT\output_masks');
                dir_output_mask = fullfile(dir_video,dir_sub_SUNS_TUnCaT,'output_masks');
                load(fullfile(dir_output_mask, ['Output_Masks_', Exp_ID, '.mat']), 'Masks');
                FinalMasks = permute(Masks,[3,2,1]);
                
            case 2 % SUNS without FISSA
                dir_resave = fullfile(dir_video,'complete_FISSA\output_masks');
                dir_output_mask = fullfile(dir_video,dir_sub_SUNS_FISSA,'output_masks');
                load(fullfile(dir_output_mask, ['Output_Masks_', Exp_ID, '.mat']), 'Masks');
                FinalMasks = permute(Masks,[3,2,1]);
                
            case 3 % CNMF-E
                dir_resave = fullfile(dir_video,'CNMFE\output_masks');
                record = Table_time_ext_CNMFE(cv,:);
                rbg = record(1);
                nk = record(2);
                rdmin = record(3);
                min_corr = record(4);
                min_pnr = record(5);
                merge_thr = record(6);
                mts = record(7);
                mtt = record(8);
                thb = record(9);
                dir_sub_CNMFE = sprintf('gSiz=%d,rbg=%0.1f,nk=%d,rdmin=%0.1f,mc=%0.2f,mp=%d,mt=%0.2f,mts=%0.2f,mtt=%0.2f',...
                    gSiz,rbg,nk,rdmin,min_corr,min_pnr,merge_thr,mts,mtt);
                dir_output_mask = fullfile(dir_video,'CNMFE',dir_sub_CNMFE);
                load(fullfile(dir_output_mask, [Exp_ID, '_Masks_',num2str(thb),'.mat']), 'Masks3');
                FinalMasks = logical(Masks3);
                
            case 4 % MIN1PIPE
                dir_resave = fullfile(dir_video,'min1pipe\output_masks');
                record = Table_time_ext_MIN1PIPE(cv,:);
                pix_select_sigthres = record(1);
                pix_select_corrthres = record(2);
                merge_roi_corrthres = record(3);
                dt = record(4);
                kappa = record(5);
                se = record(6);
                thb = record(7);
%                 dir_sub_MIN1PIPE = sprintf('pss=%0.2f_psc=%0.2f_mrc=%0.2f',...
%                     pix_select_sigthres, pix_select_corrthres, merge_roi_corrthres);
                dir_sub_MIN1PIPE = sprintf('pss=%0.2f_psc=%0.2f_mrc=%0.2f_dt=%0.2f_kappa=%0.2f_se=%d',...
                    pix_select_sigthres, pix_select_corrthres, merge_roi_corrthres, dt, kappa, se);
                dir_output_mask = fullfile(dir_video,'min1pipe',dir_sub_MIN1PIPE);
                load(fullfile(dir_output_mask, [Exp_ID, '_Masks_',num2str(thb),'.mat']), 'Masks3');
                FinalMasks = logical(Masks3);
        end
        if ~ exist(dir_resave,'dir')
            mkdir(dir_resave);
        end
        save(fullfile(dir_resave, ['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
    end
end
