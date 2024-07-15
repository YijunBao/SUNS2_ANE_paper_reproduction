clear
%load F1s.mat
filenames = {'bma22_epm_part11','bma22_epm_part12',...
    'bma22_epm_part21','bma22_epm_part22'};
radii = [6:1:10];
SNRs = [7:4:27];
[list_Recall, list_Precision, list_F1] = deal(zeros(length(filenames),length(radii),length(SNRs)));
parfor n = 1:(length(filenames)*length(radii)*length(SNRs))
	[p,q,r] = ind2sub([length(filenames) length(radii) length(SNRs)],n)
	tempname = ['./' filenames{p} '_radius' num2str(radii(q)) '_SNR' num2str(SNRs(r))  '.mat']
		if isfile(tempname)
            		B = load(tempname);
            		[recall precision F1] = GetPerformance_Jaccard('../GT Masks',filenames{p}, B.output.spatial_weights>0.2, 0.5);
            		list_Recall(n) = recall;
            		list_Precision(n) = precision;
            		list_F1(n) = F1;
		end
end
% dataset1 = dataset;
% dataset1 = reshape(dataset1,[length(filenames) length(radii) length(SNRs)])
%%
indq = zeros(length(filenames),1);
indr = zeros(length(filenames),1);
crossval_F1 = zeros(length(filenames),1);
for p = 1:length(filenames)
    temp_inds = setdiff(1:length(filenames),p);
    mean_F1s = squeeze(mean(list_F1(temp_inds,:,:),1));
    temp_max = max(mean_F1s(:));
    [tempindq,tempindr] = find(mean_F1s == temp_max)
    crossval_F1(p) = list_F1(p,tempindq,tempindr);
    indq(p) = tempindq; %radius index q
    indr(p) = tempindr; %snr index r
end

load('CV_times.mat')
save eval.mat list_Recall list_Precision list_F1 indq indr CV_times radii SNRs

