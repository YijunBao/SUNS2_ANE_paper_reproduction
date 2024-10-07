%% Generate a table of optimal parameters and F1 for 1 photon
clear
% dir_video = 'E:\1photon-small\added_refined_masks';
% dir_output_info='data_TUnCaT';
% num_Exp = 9;
% dir_output_info='data_TENASPIS_original';
% dir_output_info='data_TENASPIS';
% num_Exp = 8;
% dir_output_info='data_simulation';
dir_output_info='data_simulation_noCV\lowBG=5e+03,poisson=0.3';
num_Exp = 10;
% dir_output_info='data_CNMFE\blood_vessel_10Hz';
% num_Exp = 4;
% dir_output_info='data_CNMFE\PFC4_15Hz';
% num_Exp = 4;
% dir_output_info='data_CNMFE\bma22_epm';
% num_Exp = 4;
% dir_output_info='data_CNMFE\CaMKII_120_TMT Exposure_5fps';
% num_Exp = 4;

eval = load(fullfile(dir_output_info,'eval.mat'));
list_Precision=eval.list_Precision;
list_Recall=eval.list_Recall;
list_F1=eval.list_F1;
radii=eval.radii;
SNRs=eval.SNRs;
indq=eval.indq;
indr=eval.indr;
CV_times = eval.CV_times;

Recall_mean=squeeze(mean(list_Recall,1));
Precision_mean=squeeze(mean(list_Precision,1));
F1_mean=squeeze(mean(list_F1,1));
Time_mean=squeeze(mean(CV_times,1));

% if isvector(F1_mean)
%     F1_mean = F1_mean';
% end
[max_F1, ind_max] = max(F1_mean(:));
[L1, L2] = size(F1_mean);
[q, r] = ind2sub([L1, L2],ind_max);
disp([radii(q), SNRs(r),max_F1])
fprintf('\b');
if L1>1
    if q == 1
        disp('Decrease radius');
    elseif q == L1
        disp('Increase radius');
    end
end
if r == 1
    disp('Decrease SNR');
elseif r == L2
    disp('Increase SNR');
end

radius_all=radii(q)*ones(num_Exp,1);
SNR_all=SNRs(r)*ones(num_Exp,1);
Recall_all=squeeze(list_Recall(:,q,r));
Precision_all=squeeze(list_Precision(:,q,r));
F1_all=squeeze(list_F1(:,q,r));
Time_all=squeeze(CV_times(:,q,r));

Params_all=[radius_all, SNR_all, Recall_all, Precision_all, F1_all, Time_all]; % 
Params_all_ext=[Params_all;nanmean(Params_all,1);nanstd(Params_all,1,1)]; %([1,2,4:10],:)
disp([mean(Recall_all),mean(Precision_all),mean(F1_all)])
