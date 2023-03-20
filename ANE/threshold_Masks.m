function Masksb = threshold_Masks(Masks, thb)
    N = size(Masks,3);
    Masksb = logical(Masks);
    for k = 1:N
        mask = Masks(:,:,k);
        thred_inten = thb*max(mask,[],'all');
        Masksb(:,:,k) = threshold_frame(mask, thred_inten);
    end
