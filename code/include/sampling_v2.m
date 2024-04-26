function x_par_new = sampling_v2(x_par_old, w_par, Ns)
% x_par_old : old particle state [N x Ns]
% w_par : particle weight [1 x Ns]
% x_par_new : new particle state [N x Ns]

    % number of states
    N = size(x_par_old, 1);
    x_par_new = nan(N, Ns);
    
    % figure; 
    parfor ii = 1 : N
        % calculate kernel density using particle and weights
        [f_short, x_short] = ksdensity(x_par_old(ii, :), linspace(min(x_par_old(ii, :)), max(x_par_old(ii, :)), 100), 'Weights', w_par, 'Function', 'pdf');
        F_short = cumsum(f_short)/sum(f_short);

        % subplot(N, 1, ii); grid on; hold on;
        % plot(x_short, f_short, '.-');
        % [~, map_index] = max(f_short);
        % xline(x_short(map_index), 'b');

        above_F = (unifrnd(0, 1, 1, Ns) > F_short.');
        x_par_new(ii, :) = x_short(sum(above_F)+1);
    end
end