function [mu, sigma] = SNR_normalization(trace,meth_sigma,meth_baseline) 
% Possion noise based filter act on the traces, and calculate the median and
% std of the filterd trace
% Inputs:
    % trace is the raw trace. Each row is a timeseris of a neuron. Each column is a time frame.
    % meth_sigma is the method to determine sigma.
        % 'std':  the std of the entire trace. Has large error.
        % 'median-based std': median based std, Eq. (3) in Szymanska_2016.
        % 'quantile-based std': quantile based std.
        % 'psd': High frequency part of power spectral density.
    % meth_baseline is the method to determine sigma.
        % 'median': median of the entire trace.
        % 'ksd': mode of kernal smoothing density estimation. Better used together with 'psd'.
% Outputs:
    % mu is the baseline of the filtered trace.
    % sigma is the noise level (standard deviation of background fluorescence) of filtered trace.

%%
if diff(size(trace))<0
    trace=trace';
end

%% Determine mu of filtered trace
if exist('meth_baseline','var') && strcmp(meth_baseline,'ksd')
    [ncells,~] = size(trace);
    mu = zeros(ncells,1);
    for n = 1:ncells
        trace1 = trace(n,:);
        [f,xi] = ksdensity(trace1);
%         [~,pos] = max(f); 
%         mu(n) = xi(pos);
        maxf = max(f); 
        pos = find(f==maxf);
        mu(n) = mean(xi(pos));
    end
else
    mu = median(trace,2);
end

%% Determine sigma of the filtered video
switch meth_sigma
    case 'std'
        sigma = std(trace,1,2);
    case 'median-based std'
        AbsoluteDeviation = abs(trace - mu); 
        sigma = median(AbsoluteDeviation,2)/(sqrt(2)*erfinv(1/2)); 
    case 'quantile-based std'
        Q12 = quantile(trace, [0.25, 0.5], 2); 
        sigma = (Q12(:,2)-Q12(:,1))/(sqrt(2)*erfinv(1/2)); 
    case 'quantile-based std comp'
        pct_min = mean(abs(trace - min(trace,[],2)) < eps('single'),2);
        prob = (0.5-pct_min)*2;
        prob(prob<0) = 0;
        prob(prob>0.5) = 0.5;
        Q12 = quantile(trace, [0.25, 0.5], 2); 
        sigma = (Q12(:,2)-Q12(:,1))./(sqrt(2)*erfinv(prob)); 
        sigma(isnan(sigma))=0;
    case 'psd'
        sigma = noise_PSD(trace, 2/3);
end
end

