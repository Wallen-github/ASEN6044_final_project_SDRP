
delay_error_grid = -4 : 0.1 : 1.1;
range_grid = 0.05 : 0.05 : 3;
alpha_grid = 0.24 : 0.04 : 2;

Z_LUT = nan(length(delay_error_grid), length(range_grid), length(alpha_grid));
h = waitbar(0, "LUT generating");
for range_index = 1 : length(range_grid)
    range_this = range_grid(range_index);
    for alpha_index = 1 : length(alpha_grid)
        alpha_this = alpha_grid(alpha_index);
        cg = alpha_this/range_this/gamma(1/alpha_this);
        g = @(x) exp(-(x/range_this).^alpha_this);
        Z_LUT_local = nan(size(delay_error_grid));
            parfor delay_error_index = 1 : length(delay_error_grid)
                delay_error_this = delay_error_grid(delay_error_index);
                R_g_left = @(x) (1 + (x+delay_error_this)) * cg .* g(x) .* (x>0);
                R_g_right = @(x) (1 - (x + delay_error_this)) * cg .* g(x) .* (x>0);
                Z_LUT_local(delay_error_index) = integral(R_g_left, -delay_error_this-1, -delay_error_this) ...
                                               + integral(R_g_right, -delay_error_this, -delay_error_this+1);
            end
        Z_LUT(:, range_index, alpha_index) = Z_LUT_local;
        if mod(alpha_index, 20) == 0
            waitbar(((range_index-1)*length(alpha_grid)+alpha_index)/length(range_grid)/length(alpha_grid), h);
        end
    end
end
delete(h);

delay_error_extension = 1.2:0.1:4;
delay_error_grid = [delay_error_grid, delay_error_extension];
Z_LUT = cat(1, Z_LUT, zeros(length(delay_error_extension), length(range_grid), length(alpha_grid)));

save("Z_LUT.mat", "Z_LUT", "delay_error_grid", "range_grid", "alpha_grid");

%% display the scattered waveform

figure; grid on; hold on;
plot(delay_error_grid, reshape(Z_LUT(:, 1:5:end, 40), length(delay_error_grid), []), '.-');
plot(-1.2:0.01:1, interp1(delay_error_grid, reshape(Z_LUT(:, 1:5:end, 40), length(delay_error_grid), []), -1.2:0.01:1, 'cubic'), '-');
% legend("\alpha = " + num2str(alpha_grid.', '%.2f'));
legend("a = " + num2str(range_grid(1:10:end).', '%.2f'));