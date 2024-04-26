function Z = R_BPSK(delay_error)
    % output the ACF of BPSK signal
    % delay_error:  true - estimate (chip), any size

    Z = max(1-abs(delay_error), 0);

end

