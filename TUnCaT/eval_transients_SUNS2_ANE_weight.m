clear;
addpath(genpath('../ANE'))
dir_SUNS_sub = fullfile('SUNS_TUnCaT_SF25','4816[1]th10');
% dir_SUNS_sub = fullfile('SUNS_FISSA_SF25','4816[1]th3');
%%
dir_video='E:\1photon-small\added_refined_masks\';
list_Exp_ID = { 'c25_59_228','c27_12_326','c28_83_210',...
                'c25_163_267','c27_114_176','c28_161_149',...
                'c25_123_348','c27_122_121','c28_163_244'};
% list_Exp_ID = list_Exp_ID(1:5);

dir_SUNS = fullfile(dir_video, dir_SUNS_sub);

list_spike_type = {'include'}; % {'only','include','exclude'};
% spike_type = 'exclude'; % {'include','exclude','only'};
list_sigma_from = {'Unmix'}; % {'Raw','Unmix'}; 
dir_label = fullfile(dir_video,'GT transients');
dir_GT_masks = fullfile(dir_video,'GT Masks'); % FinalMasks_
dir_traces_GT = fullfile(dir_GT_masks, 'TUnCaT/raw');

method = 'SUNS2-ANE+TUnCaT'; % 'SUNS2-ANE+TUnCaT' % 
if contains(method,'ANE')
    dir_traces = fullfile(dir_SUNS, 'output_masks/add_new_blockwise/trained dropout 0.8exp(-15)/TUnCaT_time');
else
    dir_traces = fullfile(dir_SUNS, 'output_masks/TUnCaT_time');
end
% dir_traces = [dir_traces,'_1e-2'];
dir_traces = [dir_traces,'_tol=1e-4_bin=2'];
% list_thred_ratio = 7;

list_video={'Raw'}; % {'Raw','SNR'}
% addon = '';  % ''; % '_eps=0.1'; % _n_iter
% max_alpha = inf;
list_baseline_std = {''}; % '_ksd-psd', 

%%
for bsid = 1:length(list_baseline_std)
    baseline_std = list_baseline_std{bsid};
% baseline_std = '_ksd-psd'; % ''; % '_psd'; % 
if contains(baseline_std,'psd')
    std_method = 'psd';  % comp
    if contains(baseline_std,'ksd')
        baseline_method = 'ksd';
    else
        baseline_method = 'median';
    end
else
    std_method = 'quantile-based std comp';
    baseline_method = 'median';
end

for tid = 1:length(list_spike_type)
    spike_type = list_spike_type{tid}; % 
    for inds = 1:length(list_sigma_from)
        sigma_from = list_sigma_from{inds};
        for ind_video = 1:length(list_video)
            video = list_video{ind_video};
            if contains(baseline_std, 'psd')
                if contains(video,'SNR')
                    list_thred_ratio=100:20:300;% jGCaMP7c; % 0.3:0.3:3; % 1:0.5:6; % 
                else
                    list_thred_ratio=100:20:300; % jGCaMP7c; % 0.3:0.3:3; % 1:0.5:6; % 
                end
            else
                list_thred_ratio=4:12; % 6:0.5:9; % 8:2:20; % 
            end
%             folder = sprintf('traces_%s_%s%s',method,video,addon);
%             disp(folder);
%             dir_traces = fullfile(dir_video,folder);
            useTF = strcmp(video, 'Raw');

%             list_alpha = 1; %
            num_Exp=length(list_Exp_ID);
            num_ratio=length(list_thred_ratio);
            [list_recall, list_precision, list_F1, list_nTP1, list_nFP, list_nTP2, list_nFN,...
                list_recall_match, list_precision_match, list_F1_match,...
                list_nTP1_match, list_nFP_match, list_nTP2_match, list_nFN_match]=deal(zeros(num_Exp, num_ratio));
            list_unmix_time = zeros(num_Exp,1);

            if useTF
                dFF = h5read(fullfile(dir_video,'1P_spike_tempolate.h5'),'/filter_tempolate')';
                dFF = dFF(dFF>exp(-1));
                dFF = dFF'/sum(dFF);
                kernel=fliplr(dFF);
            else
                kernel = 1;
            end

            %%
            fprintf('Neuron Toolbox');
            for ii = 1:num_Exp
                Exp_ID = list_Exp_ID{ii};
                fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b%12s: ',Exp_ID);
                load(fullfile(dir_label,['output_',Exp_ID,'.mat']),'output');
                if strcmp(spike_type,'exclude')
                    for oo = 1:length(output)
                        if ~isempty(output{oo}) && all(output{oo}(:,3))
                            output{oo}=[];
                        end
                    end
                elseif strcmp(spike_type,'only')
                    for oo = 1:length(output)
                        if ~isempty(output{oo}) && ~all(output{oo}(:,3))
                            output{oo}=[];
                        end
                    end
                end

                load(fullfile(dir_traces_GT,[Exp_ID,'.mat']),'traces', 'bgtraces');
                if useTF
                    traces_GT_filt=conv2(traces'-bgtraces',kernel,'valid');
                else
                    traces_GT_filt=traces'-bgtraces';
                end
                [mu_GT, sigma_GT] = SNR_normalization(traces_GT_filt,std_method,baseline_method);
                traces_GT_norm = (traces_GT_filt - mu_GT)./sigma_GT;

                load(fullfile(dir_traces,'raw',[Exp_ID,'.mat']),'traces', 'bgtraces');
                if useTF
                    traces_raw_filt=conv2(traces'-bgtraces',kernel,'valid');
                else
                    traces_raw_filt=traces'-bgtraces';
                end
                [mu_raw, sigma_raw] = SNR_normalization(traces_raw_filt,std_method,baseline_method);
                traces_out_norm = (traces_raw_filt - mu_raw)./sigma_raw;

                load(fullfile(dir_traces,['Output_Masks_',Exp_ID,'.mat']),'Masks','traces','TUnCaT_time');
                list_unmix_time(ii) = TUnCaT_time;
                Masks3 = Masks;
                traces_unmixed = traces';
                if useTF
                    traces_unmixed_filt=conv2(traces_unmixed,kernel,'valid');
                else
                    traces_unmixed_filt=traces_unmixed;
                end
                [mu_unmixed, sigma_unmixed] = SNR_normalization...
                    (traces_unmixed_filt,std_method,baseline_method);

        %         fprintf('Neuron Toolbox');
                if strcmp(sigma_from,'Raw')
                    sigma = sigma_raw;
                else
                    sigma = sigma_unmixed;
                end
                
                [SpatialRecall, SpatialPrecision, SpatialF1,m] = GetPerformance_Jaccard(...
                    dir_GT_masks,Exp_ID,Masks3,0.5);
                [select_seg, select_GT] = find(m');
                n_match = length(select_GT);

                parfor kk=1:num_ratio
                    thred_ratio=list_thred_ratio(kk);
        %             fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\bthresh=%5.2f: ',num_ratio);
        %             thred=mu_unmixed+sigma_unmixed*thred_ratio;
                    [recall_match, precision_match, F1_match,individual_recall,individual_precision,spikes_GT_array_match,spikes_eval_array_match]...
                        = GetPerformance_SpikeDetection_split_weight(output(select_GT),traces_unmixed_filt(select_seg,:),...
                            thred_ratio,sigma(select_seg),mu_unmixed(select_seg),traces_GT_norm(select_GT,:));
                    list_recall_match(ii,kk)=recall_match; 
                    list_precision_match(ii,kk)=precision_match;
                    list_F1_match(ii,kk)=F1_match;
                    list_nTP1_match(ii,kk) = sum(cellfun(@(x) sum(x(:,3)), spikes_GT_array_match(~cellfun(@isempty,spikes_GT_array_match))));
                    list_nTP2_match(ii,kk) = sum(cellfun(@(x) sum(x(:,3)), spikes_eval_array_match(~cellfun(@isempty,spikes_eval_array_match))));
                    list_nFN_match(ii,kk) = sum(cellfun(@(x) sum(~x(:,3)), spikes_GT_array_match(~cellfun(@isempty,spikes_GT_array_match))));
                    list_nFP_match(ii,kk) = sum(cellfun(@(x) sum(~x(:,3)), spikes_eval_array_match(~cellfun(@isempty,spikes_eval_array_match))));

                    [recall, precision, F1,spikes_GT_array,spikes_eval_array]...
                        = GetPerformance_SpikeDetection_unmatch_weight(output,traces_unmixed_filt,m,thred_ratio,sigma,mu_unmixed, traces_GT_norm, traces_out_norm);
                    list_recall(ii,kk)=recall; 
                    list_precision(ii,kk)=precision;
                    list_F1(ii,kk)=F1;
                    list_nTP1(ii,kk) = sum(cellfun(@(x) sum(x(:,3)), spikes_GT_array(~cellfun(@isempty,spikes_GT_array))));
                    list_nTP2(ii,kk) = sum(cellfun(@(x) sum(x(:,3)), spikes_eval_array(~cellfun(@isempty,spikes_eval_array))));
                    list_nFN(ii,kk) = sum(cellfun(@(x) sum(~x(:,3)), spikes_GT_array(~cellfun(@isempty,spikes_GT_array))));
                    list_nFP(ii,kk) = sum(cellfun(@(x) sum(~x(:,3)), spikes_eval_array(~cellfun(@isempty,spikes_eval_array))));
                end
            end
            fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b');

            %         if num_ratio>1
            %             figure; plot(list_thred_ratio,[list_recall,list_precision,list_F1],'LineWidth',2);
            %             legend('Recall','Precision','F1');
            %             xlabel('thred_ratio','Interpreter','none');
            %             [~,ind]=max(list_F1);
            %             fprintf('\nRecall=%f\nPrecision=%f\nF1=%f\nthred_ratio=%f\n',list_recall(ind),list_precision(ind),list_F1(ind),list_thred_ratio(ind));
            %         else
            %             fprintf('\nRecall=%f\nPrecision=%f\nF1=%f\nthred_ratio=%f\n',recall, precision, F1,thred_ratio);
            %         end
           %%
            if num_ratio == 1
                Table_unmatch = [list_nTP1, list_nFN, list_recall, list_nTP2, ...
                    list_nFP, list_precision, list_F1, list_unmix_time];
                Table_unmatch_ext = [Table_unmatch;mean(Table_unmatch);std(Table_unmatch,1,1)];
                Table_match = [list_nTP1_match, list_nFN_match, list_recall_match, list_nTP2_match, ...
                    list_nFP_match, list_precision_match, list_F1_match, list_unmix_time];
                Table_match_ext = [Table_match;mean(Table_match);std(Table_match,1,1)];
            end
            save(fullfile(dir_traces,sprintf('Transient_accuracy_weight_%s.mat',method)),...
                'list_recall_match','list_precision_match','list_F1_match','list_unmix_time',...
                'list_recall','list_precision','list_F1','list_thred_ratio');
%             save(sprintf('%s\\scores_alpha=1_%s_%sVideo_%sSigma%s.mat',...
%                 spike_type,method,video,sigma_from,baseline_std),...
%                 'list_recall','list_precision','list_F1','list_thred_ratio','list_alpha');
%             save(sprintf('scores_ours_%sVideo_%sSigma (tol=1e-4, max_iter=%d).mat',video,sigma_from,max_iter),...
%                 'list_recall','list_precision','list_F1','list_thred_ratio','list_alpha');
            mean_F1 = squeeze(mean(list_F1,1))';
            [max_F1, ind] = max(mean_F1(:));
            disp([list_thred_ratio(ind),max_F1])
            mean_F1_match = squeeze(mean(list_F1_match,1))';
            [max_F1_match, ind_match] = max(mean_F1_match(:));
            disp([list_thred_ratio(ind_match),max_F1_match])
            fprintf('\b');
%             if ind1 == 1
%                 disp('Decrease alpha');
%             elseif ind1 == L1
%                 disp('Increase alpha');
%             end
            if num_ratio > 1
                if ind == 1
                    disp('Decrease thred_ratio');
                elseif ind == num_ratio
                    disp('Increase thred_ratio');
                end
            end
        end
    end
end
end