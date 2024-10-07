%% Generate a table of optimal parameters and F1 for 1 photon
clear
% dir_output_info='data_TUnCaT';
% num_Exp = 9;
% dir_output_info='data_TENASPIS_original';
% dir_output_info='data_TENASPIS';
% num_Exp = 8;
dir_output_info='data_simulation';
num_Exp = 10;
% dir_output_info='data_CNMFE\blood_vessel_10Hz';
% num_Exp = 4;
% dir_output_info='data_CNMFE\PFC4_15Hz';
% num_Exp = 4;
% dir_output_info='data_CNMFE\bma22_epm';
% num_Exp = 4;
% dir_output_info='data_CNMFE\CaMKII_120_TMT Exposure_5fps';
% num_Exp = 4;
% list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
%     'c25_163_267','c27_114_176','c28_161_149',...
%     'c25_123_348','c27_122_121','c28_163_244'};
[radius_all, SNR_all, F1_all, Recall_all, Precision_all, Time_all, ...
    F1_mean, Recall_mean, Precision_mean, Time_mean] = deal(zeros(num_Exp,1));
Table_max = zeros(num_Exp,3);

eval = load(fullfile(dir_output_info,'eval.mat'));
list_Precision=eval.list_Precision;
list_Recall=eval.list_Recall;
list_F1=eval.list_F1;
radii=eval.radii;
SNRs=eval.SNRs;
indq=eval.indq;
indr=eval.indr;
CV_times = eval.CV_times;

for CV=1:num_Exp %[1,2,4:10] %
%     Exp_ID=list_Exp_ID{CV};
    q = indq(CV);
    r = indr(CV);
    train = setdiff(1:num_Exp,CV);
    radius_all(CV)=radii(q);
    SNR_all(CV)=SNRs(r);
    Recall_all(CV)=list_Recall(CV,q,r);
    Precision_all(CV)=list_Precision(CV,q,r);
    F1_all(CV)=list_F1(CV,q,r);
    Time_all(CV)=CV_times(CV,q,r);
    Recall_mean(CV)=mean(list_Recall(train,q,r));
    Precision_mean(CV)=mean(list_Precision(train,q,r));
    F1_mean(CV)=mean(list_F1(train,q,r));
    Time_mean(CV)=mean(CV_times(train,q,r));
end
    
Params_all=[radius_all, SNR_all, Recall_all, Precision_all, F1_all, Time_all, ...
    Recall_mean, Precision_mean, F1_mean, Time_mean]; 
Params_all_ext=[Params_all;nanmean(Params_all,1);nanstd(Params_all,1,1)]; %([1,2,4:10],:)
disp([mean(Recall_all),mean(Precision_all),mean(F1_all)])
