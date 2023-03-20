function noise = noise_PSD(traces_input, pct)
    % pct = 2/3;
    traces_raw = traces_input - mean(traces_input,2);
    T = size(traces_raw,2);
    spectrum = fft(traces_raw,[],2);
    spectrum(:,1) = 0;
%     power = abs(traces_raw).^2;
    PSD = abs(spectrum).^2/T;
%     plot(sum(PSD,2)./sum(power,2));

    above = round(pct*T/2);
    range = above:(T-above);
    noise = sqrt(mean(PSD(:,range),2));
end
% figure; histogram(noise,5:0.5:18);
% figure; histogram(sigma_raw,5:0.5:18);
