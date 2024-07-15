clear
%load F1s.mat
filenames = {'blood_vessel_10Hz_part11','blood_vessel_10Hz_part12','blood_vessel_10Hz_part21','blood_vessel_10Hz_part22'};

radii = [3:7];
SNRs = [7:4:23];
dataset = zeros(length(filenames),length(radii),length(SNRs));
parfor n = 1:(length(filenames)*length(radii)*length(SNRs))
	[p,q,r] = ind2sub([length(filenames) length(radii) length(SNRs)],n)
	tempname = ['./' filenames{p} '_radius' num2str(radii(q)) '_SNR' num2str(SNRs(r))  '.mat']
		if isfile(tempname)
            		B = load(tempname);
            		[recall precision F1] = GetPerformance_Jaccard('../GT Masks',filenames{p}, B.output.spatial_weights>0.2, 0.5);
            		dataset(n) = F1;
		end
end
dataset1 = dataset;
dataset1 = reshape(dataset1,[length(filenames) length(radii) length(SNRs)])
save F1s.mat dataset dataset1
%%
load F1s.mat
indq = zeros(length(filenames),1);
indr = zeros(length(filenames),1);
crossval_F1 = zeros(length(filenames),1);
for p = 1:length(filenames)
    temp_inds = setdiff(1:length(filenames),p);
    mean_F1s = squeeze(mean(dataset(temp_inds,:,:),1));
    temp_max = max(mean_F1s(:));
    [tempindq,tempindr] = find(mean_F1s == temp_max)
    crossval_F1(p) = dataset(p,tempindq,tempindr);
    indq(p) = tempindq; %radius index q
    indr(p) = tempindr; %snr index r
end
save F1s.mat dataset dataset1 indq indr
