function [onlyone, good_amp, list_peak_locations]=isolate_transients_d_nomix...
    (d, ROIs2, d_min, d_max, MPP, before, after)
% Function to calculate decay time and average spike shape from a moderate SNR range.
% Inputs: 
%     d: traces of each neuron. length = (ncells, T)
%     ROIs2: neuron masks in 2D format. length = (Lx*Ly, ncells)
%     d_min: minimum peak SNR value
%     d_max: maximum peak SNR value
%     before: number of frames before spike peak
%     after: number of frames after spike peak
% Outputs:

[ncells,T]=size(d);
% fine_ratio=1;
% MPP = 1; % d_min/2;
% if ncells>T
%     traces_raw=traces_raw';
%     [ncells,T]=size(traces_raw);
% end

% list_spikes=cell(ncells,1);
% F0=median(traces_raw,2);
% if min(F0)<=0
%     warning('Please input raw trace rather than df/f or SNR');
% end
% sigma = median(abs(traces_raw-F0),2)/(sqrt(2)*erfinv(1/2)); 

onlyone=d>d_min;
% Remove the spikes that are coactivate with its neighbours. 
for tt=1:T
    active=find(onlyone(:,tt));
    sum_masks=sum(ROIs2(:,active),2);
    overlap=sum_masks>1;
    if any(overlap,'all')
        for aa=active
            if any(ROIs2(:,aa) & overlap,'all')
                onlyone(aa,tt)=0;
            end
        end
    end
end

% Select the spikes whose peak SNR is between d_min and d_max
good_amp=(d>d_min & d<d_max);
good_amp_diff=diff(padarray(good_amp,[0,1],false,'both'),1,2);
list_peak_locations = cell(ncells,1);

for nn=1:ncells
    good_start=find(good_amp_diff(nn,:)==1);
    good_end=find(good_amp_diff(nn,:)==-1)-1;
    if isempty(good_start)
        continue;
    end
    num_good=length(good_start);
    low_between=false(1,num_good+1);
    sep=1:good_start(1)-1;
    % low_between is used to ensure sufficient temporal distance between two spikes
    low_between(1)=~sum(d(nn,sep)>d_max);
    for pp=2:num_good
        sep=good_end(pp-1)+1:good_start(pp)-1;
        low_between(pp)=~sum(d(nn,sep)>d_max);
    end
    sep=good_end(num_good)+1:T;
    low_between(num_good+1)=~sum(d(nn,sep)>d_max);
    good_amp_dist=[good_start(1)-1,good_start(2:end)-good_end(1:end-1),T-good_end(end)];
    % good_amp_dist=circshift(good_amp_start,[-1,0])-good_amp_end;
    good_amp_dist=good_amp_dist.*low_between;
%     good_dist=(good_amp_dist(1:end-1)>=before+after) & (good_amp_dist(2:end)>=before+after);
    good_dist=(good_amp_dist(1:end-1)>=1) & (good_amp_dist(2:end)>=1);
    good_ind=find(good_dist);
    num_good=length(good_ind);
    
%     spikes=zeros(num_good,1+before+after);
    location_all = [];
    for jj=1:num_good
%         pp=0;
        spike_current=d(nn,good_start(good_ind(jj))-1:good_end(good_ind(jj))+1);
        [peaks, locations]=findpeaks(spike_current,'MinPeakProminence',MPP);
        if ~isempty(peaks) % && length(peaks)==1
%                 pp=pp+1;
            location_all=[location_all,good_start(good_ind(jj))-2+locations];
%                 spikes(pp,:)=d(nn,location_all-before:location_all+after)/peaks; %
        end
    end
        
    onlyone_peaks = false(1,length(location_all));
    for kk = 1:length(location_all)
        location = location_all(kk);
%         onlyone_current=onlyone(nn,max(1,location-before):min(T,location+after));
%         onlyone_peaks(kk) = all(onlyone_current);
        onlyone_peaks(kk) = onlyone(nn,location);
    end
    list_peak_locations{nn}=location_all(onlyone_peaks);
end
