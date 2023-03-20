function [C, f, A, seeds, datasmth, cutoff, pkcutoff, Pnew, S, YrA] = refine_sig(m, A, b, Cin, fin, seeds, datasmth, cutoff, pkcutoff, p, options)
% [C, f, Pnew, S, YrA] = refine_sig refine signal by CNMF
%   modified from E Pnevmatikakis
%   Jinghao Lu 06/10/2016

    hsig = tic;
    %% initialization %%
    defoptions = CNMFSetParms;
    if ~isfield(options,'temporal_iter') || isempty(options.temporal_iter)
        ITER = defoptions.temporal_iter; 
    else
        ITER = options.temporal_iter; 
    end           % number of block-coordinate descent iterations
    
    if ~isfield(options,'temporal_parallel')
        options.temporal_parallel = defoptions.temporal_parallel; 
    end

    [d1, d2, T] = size(m, 'reg');
    K = size(A, 2);
    A = [A, b];
    S = zeros(size(Cin));
    Cin = [Cin; fin];
    C = Cin;
    AA = (A' * A) ./ sum(A, 1);
    
    nsize = d1 * d2 * T * 8 * 2; %%% size of double %%%
    nbatch = batch_compute(nsize);
    ebatch = ceil(T / nbatch);
    idbatch = [1: ebatch: T, T + 1];
    nbatch = length(idbatch) - 1;
    YrA = zeros(T, K + 1);
    for ib = 1: nbatch
        idtmp = idbatch(ib): idbatch(ib + 1) - 1;
        ytmp = m.reg(1: d1, 1: d2, idtmp);
        ytmp = reshape(ytmp, d1 * d2, []);
        tominus = A * Cin(:, idtmp);
%         tominus = zeros(d1 * d2, size(ytmp, 2));
%         for i = 1: K
%             cid = A(:, i) > 0;
%             ctmp = A(cid, i) * Cin(i, idtmp);
%             tominus(cid, :) = max(tominus(cid, :), ctmp);
%             disp(num2str(i))
%         end
%         tominus = tominus + b * fin(idtmp);
        yrt = ytmp - tominus;
        YrA(idtmp, :) = yrt' * A ./ sum(A, 1);
    end
    
%     yat = zeros(T, size(A, 2));
%     for i = 1: nbatch
%         tmp = m.reg(1: d1, 1: d2, idbatch(i): idbatch(i + 1) - 1);
%         tmp = double(reshape(tmp, d1 * d2, (idbatch(i + 1) - idbatch(i))));
%         yat(idbatch(i): idbatch(i + 1) - 1, :) = tmp' * A;
%     end
%     YA = yat / spdiags(nA(:), 0, length(nA), length(nA));
%     YrA = (YA - Cin' * AA);
    
    Pnew.gn = cell(K, 1);
    Pnew.b = cell(K, 1);
    Pnew.c1 = cell(K, 1);
    Pnew.neuron_sn = cell(K, 1);

    %% time series refinement %%
    if options.temporal_parallel
        for iter = 1: ITER
            [O,lo] = update_order(A(:, 1: K));
            for jo = 1: length(O)
                Ytemp = YrA(:, O{jo}(:)) + Cin(O{jo}, :)';
                Ctemp = zeros(length(O{jo}), T);
                Stemp = zeros(length(O{jo}), T);
                btemp = zeros(length(O{jo}), 1);
                sntemp = btemp;
                c1temp = btemp;
                gtemp = cell(length(O{jo}), 1);
                % FN added the part below in order to save SAMPLES as a field of P
                parfor jj = 1:length(O{jo})
                    if p == 0   % p = 0 (no dynamics assumed)
                        cc = max(Ytemp(:, jj), 0);
                        Ctemp(jj, :) = full(cc');
                        Stemp(jj, :) = Ctemp(jj, :);
                    else
                        [cc, cb, c1, gn, sn, spk] = constrained_foopsi(Ytemp(:, jj), [], [], [], [], options);
                        gd = max(roots([1, -gn']));  % decay time constant for initial concentration
                        gd_vec = gd .^ ((0: T-1));
                        Ctemp(jj, :) = full(cc(:)' + cb + c1 * gd_vec);
                        Stemp(jj, :) = spk(:)';
                        Ytemp(:, jj) = Ytemp(:, jj) - Ctemp(jj, :)';
                        btemp(jj) = cb;
                        c1temp(jj) = c1;
                        sntemp(jj) = sn;
                        gtemp{jj} = gn(:)';
                    end
                end
                if p > 0
                    Pnew.b(O{jo}) = num2cell(btemp);
                    Pnew.c1(O{jo}) = num2cell(c1temp);
                    Pnew.neuron_sn(O{jo}) = num2cell(sntemp);
                    for jj = 1:length(O{jo})
                        Pnew.gn(O{jo}(jj)) = gtemp(jj);
                    end
%                     YrA = YrA - (Ctemp - C(O{jo}(:), :))' * AA(O{jo}(:), :);
                    YrA = YrA - (Ctemp - C(O{jo}(:), :))' * AA(O{jo}(:), :);
                    C(O{jo}(:), :) = Ctemp;
                    S(O{jo}(:), :) = Stemp;
                else
                    YrA = YrA - (Ctemp - C(O{jo}(:), :))' * AA(O{jo}(:), :);
                    C(O{jo}(:), :) = Ctemp;
                    S(O{jo}(:), :) = Stemp;
                end
                fprintf('%i out of %i components updated \n', sum(lo(1: jo)), K);
            end
            for ii = K + 1: size(C, 1)
                cc = full(max(YrA(:, ii)' + Cin(ii, :), 0));
                YrA = YrA - (cc - C(ii, :))' * AA(ii, :);
                C(ii,:) = cc;
            end

            if norm(Cin - C, 'fro') / norm(C, 'fro') <= 1e-3
                % stop if the overall temporal component does not change by much
                break;
            else
                Cin = C;
            end
        end
    else
        for iter = 1: ITER
            perm = randperm(K + size(b, 2));
            for jj = 1: K
                ii = perm(jj);
                if ii <= K
                    ff = find(AA(ii, :));
                    if p == 0   % p = 0 (no dynamics assumed)
                        Ytemp = YrA(:, ii) + Cin(ii, :)';
                        cc = max(Ytemp, 0);
                        YrA(:, ff) = YrA(:, ff) - (cc - C(ii, :)') * AA(ii, ff);
                        C(ii, :) = full(cc');
                        S(ii, :) = C(ii, :);
                    else
                        Ytemp = YrA(:, ii) + Cin(ii, :)';
                        [cc, cb, c1, gn, sn, spk] = constrained_foopsi(Ytemp, [], [], [], [], options);
                        Pnew.gn{ii} = gn;
                        gd = max(roots([1, -gn']));  % decay time constant for initial concentration
                        gd_vec = gd .^ ((0: T - 1));
                        YrA(:, ff) = YrA(:, ff) - (cc(:) + cb + c1 * gd_vec' - C(ii, :)') * AA(ii, ff);
                        C(ii, :) = full(cc(:)' + cb + c1 * gd_vec);
                        S(ii, :) = spk(:)';
                        Pnew.b{ii} = cb;
                        Pnew.c1{ii} = c1;
                        Pnew.neuron_sn{ii} = sn;
                    end
                else
                    Ytemp = YrA(:, ii) + Cin(ii, :)';
                    cc = max(Ytemp, 0);
                    YrA = YrA - (cc - C(ii, :)') * AA(ii, :);
                    C(ii, :) = full(cc');
                end
                fprintf('%i out of total %i temporal components updated \n', jj, K);
            end
            if norm(Cin - C, 'fro') / norm(C, 'fro') <= 1e-3
                % stop if the overall temporal component does not change by much
                break;
            else
                Cin = C;
            end
        end
    end
    f = C(K + 1: end, :);
    C = C(1: K, :);
    YrA = YrA(:, 1: K)';
    
    %%% delete empty ROIs %%%
    C= max(C, 0);
    f = max(0, f);
    inddel = union(find(sum(A, 1) == 0), find(sum(C, 2) == 0));
    A = A(:, setdiff(1: K, inddel));
    C = C(setdiff(1: K, inddel), :);
    datasmth = datasmth(setdiff(1: K, inddel), :);
    cutoff = cutoff(setdiff(1: K, inddel));
    pkcutoff = pkcutoff(setdiff(1: K, inddel));
    seeds = seeds(setdiff(1: K, inddel));
    
    time = toc(hsig);
    disp(['Done refine sig, ', num2str(time), ' seconds'])
end