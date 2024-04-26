function x = RandomWalk(x0, F, Q, N)
% initial state : x0 (nx1)
% extrapolation matrix : F (nxn)
% process noise cov matrix: Q (nxn)
% epoch number: N (including init state)

    x = nan(length(x0), N);

    % process noise
    w = mvnrnd(zeros(size(x0)), Q, N-1).';

    for ii = 1 : N
        if ii == 1
            x(:, 1) = x0;
        else
            x(:, ii) = F * x(:, ii-1) + w(:, ii-1);
        end
    end

end

