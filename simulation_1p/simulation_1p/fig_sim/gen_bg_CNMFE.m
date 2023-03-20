% clear;
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_Exp_ID = list_data_names;
eid = 1;
Exp_ID = list_Exp_ID{eid};
Exp_ID = [Exp_ID,'_pad'];
path_name = 'E:\data_CNMFE\pad videos';
dir_save = fullfile(path_name,'CNMFE');
load(fullfile(dir_save,['output_bg_info_',Exp_ID]),'output_bg_info');
d10 = 64;
d20 = 64;
dir_bg = fullfile(path_name,'CNMF-E bg_extract');
if ~exist(dir_bg,'dir')
    mkdir(dir_bg)
end

%%
for bx=2:4
    for by=2:4
        output_block = output_bg_info(bx,by);
        % load('E:\simulation_CNMFE_corr_noise\lowBG=1e+03,poisson=0.1\CNMFE\gSiz=12,rbg=1.8,nk=1,rdmin=3.0,mc=0.20,mp=2,mt=0.20,mts=0.80,mtt=0.40\sim_0_result.mat','neuron')
        % Y = h5read('E:\simulation_CNMFE_corr_noise\lowBG=1e+03,poisson=0.1\sim_0.h5','/mov');
        Y3 = output_block.Y;
        [d1,d2,t] = size(Y3);
        Y2 = reshape(Y3, d1*d2,t);
        A2 = output_block.A;
        C = output_block.C;
        Bf = double(Y2) - A2 * C;
        b02 = mean(Bf,2);
        b03 = single(reshape(b02,d1,d2));
        Bf0 = single(Bf - b02);
        W = output_block.W;
        WBf0 = single(full(W)*Bf0);
        Bf03 = reshape(Bf0,d1,d2,t);
        WBf03 = reshape(WBf0,d10,d20,t);

        patch_pos = output_block.patch_pos;
        block_pos = output_block.block_pos;
        filename = sprintf('%s_part%d%d.mat',Exp_ID,bx,by);
        save(fullfile(dir_bg,filename),'b03','Bf03','WBf03','patch_pos','block_pos');
    end
end
% figure; imagesc(Bf0); colorbar;
% figure; imagesc(WBf0); colorbar;
% figure; imagesc(WBf0 - Bf0); colorbar;
% figure; imshow3D(Bf03);
% figure; imshow3D(WBf03);
