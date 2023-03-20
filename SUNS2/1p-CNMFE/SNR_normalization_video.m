function [mu, sigma] = SNR_normalization_video(video,meth_sigma,meth_baseline) 
% Possion noise based filter act on the videos, and calculate the median and
% std of the filterd video
% Inputs:
    % video is the raw video. Each row is a timeseris of a neuron. Each column is a time frame.
    % meth_sigma is the method to determine sigma.
        % 'std':  the std of the entire video. Has large error.
        % 'median-based std': median based std, Eq. (3) in Szymanska_2016.
        % 'quantile-based std': quantile based std.
        % 'psd': High frequency part of power spectral density.
    % meth_baseline is the method to determine sigma.
        % 'median': median of the entire video.
        % 'ksd': mode of kernal smoothing density estimation. Better used together with 'psd'.
% Outputs:
    % mu is the baseline of the filtered video.
    % sigma is the noise level (standard deviation of background fluorescence) of filtered video.

%% Determine mu of filtered video
if exist('meth_baseline','var') && strcmp(meth_baseline,'ksd')
    [Lx,Ly,T] = size(video);
    mu = zeros(Lx,Ly);
    for x = 1:Lx
        for y = 1:Ly
            video1 = squeeze(video(x,y,:));
            [f,xi] = ksdensity(video1);
    %         [~,pos] = max(f); 
    %         mu(n) = xi(pos);
            maxf = max(f); 
            pos = find(f==maxf);
            mu(x,y) = mean(xi(pos));
        end
    end
else
    mu = median(video,3);
end

%% Determine sigma of the filtered video
switch meth_sigma
    case 'std'
        sigma = std(video,1,3);
    case 'median-based std'
        AbsoluteDeviation = abs(video - mu); 
        sigma = median(AbsoluteDeviation,3)/(sqrt(2)*erfinv(1/2)); 
    case 'quantile-based std'
        Q12 = quantile(video, [0.25, 0.5], 3); 
        sigma = (Q12(:,:,2)-Q12(:,:,1))/(sqrt(2)*erfinv(1/2)); 
    case 'quantile-based std comp'
        pct_min = mean(abs(video - min(video,[],3)) < eps('single'),3);
        prob = (0.5-pct_min)*2;
        prob(prob<0) = 0;
        prob(prob>0.5) = 0.5;
        Q12 = quantile(video, [0.25, 0.5], 3); 
        sigma = (Q12(:,:,2)-Q12(:,:,1))./(sqrt(2)*erfinv(prob)); 
        sigma(isnan(sigma))=0;
    case 'psd'
        sigma = noise_PSD(video, 2/3);
end
end

