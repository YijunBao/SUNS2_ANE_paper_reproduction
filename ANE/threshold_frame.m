function [frame_thred, noisy, final_thred] = threshold_frame(frame, thred_inten, target_area, unmasked)
    [Lxm,Lym] = size(frame);
    if ~exist('unmasked','var')
        unmasked = true(Lxm,Lym);
    end
    final_thred = thred_inten;
    frame_thred = frame > thred_inten;
    noisy = false;
    [L,nL] = bwlabel(frame_thred,4);
    if nL>=1
        % Find the thresholded region with the largest area
        list_area_L = zeros(nL,1);
        list_area_masked_L = zeros(nL,1);
        for ix = 1:Lxm
        for iy = 1:Lym
            val = L(ix,iy);
            if val
                list_area_L(val) = list_area_L(val)+1;
                list_area_masked_L(val) = list_area_masked_L(val)+unmasked(ix,iy);
            end
        end
        end
        [max_area] = max(list_area_L);
        if max_area < sum(frame_thred,'all')/2
            noisy = true;
        end
        [~,iLm] = max(list_area_masked_L);
        core_region = (L == iLm);
        
%         % Find the core region
%         if ~exist('unmasked','var')
%             core_region = (L == iL);
%         else
% %             frame_thred_masked = frame_thred & masked;
%             list_area_masked_L = zeros(nL,1);
%             [xx,yy] = find(unmasked);
%             for ii = 1:length(xx)
%                 val = L(xx(ii),yy(ii));
%                 if val
%                     list_area_masked_L(val) = list_area_masked_L(val)+1;
%                 end
%             end
%         end
        
        % If the area of the core region is too small, iteratively lower the threshold
        if ~exist('target_area','var') || isempty(target_area) || max_area >= target_area
            frame_thred = core_region;
        else
            frame_thred_old = frame_thred;
            while max_area < target_area % && thred_inten > 0.1 
                final_thred = thred_inten;
                frame_thred_old = frame_thred;
                if thred_inten > 1
                    thred_inten = thred_inten * 0.9;
                else
                    thred_inten = thred_inten - 0.1;
                end
    %                 thred_inten = quantile(avg_frame_use, 1-mean(mask_sub,'all')/max_area*area(nn), 'all');
                frame_thred = frame > thred_inten;
                [L,nL] = bwlabel(frame_thred,4);
                list_area_L = zeros(nL,1);
                for ix = 1:Lxm
                for iy = 1:Lym
                    val = L(ix,iy);
                    if val
                        list_area_L(val) = list_area_L(val)+1;
                    end
                end
                end
                [max_area] = max(list_area_L);
            end
            [L,nL] = bwlabel(frame_thred_old,4);
%             list_area_L = zeros(nL,1);
%             for ix = 1:Lxm
%             for iy = 1:Lym
%                 val = L(ix,iy);
%                 if val
%                     list_area_L(val) = list_area_L(val)+1;
%                 end
%             end
%             end
%             [max_area,iL] = max(list_area_L);

            list_area_L_core = zeros(nL,1);
            [xx,yy] = find(core_region);
            for ii = 1:length(xx)
                val = L(xx(ii),yy(ii));
                if val
                    list_area_L_core(val) = list_area_L_core(val)+1;
                end
            end
            [~,iL] = max(list_area_L_core);
            frame_thred = (L==iL);
        end
    end
    
    % Fill the holes in the thresholded core region
    [L0,nL0] = bwlabel(~frame_thred,4);
    if nL0>1
        list_area_L0 = zeros(nL0,1);
        for ix = 1:Lxm
        for iy = 1:Lym
            val = L0(ix,iy);
            if val
                list_area_L0(val) = list_area_L0(val)+1;
            end
        end
        end
        [~,iL0] = max(list_area_L0);
        frame_thred = (L0 ~= iL0);
    end
