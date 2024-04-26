clc;
clear;
close all;
format LONG;
addpath(genpath("./"))

%% parameters

% speed of light
c = 299792458;
% chipping rate (Hz)
fc = 1.023e6;
% chiplength (m)
lc = c / fc;

% coherent integration time
T = 20e-3;
% simulation duration
duration = 10;
% timeline
time = 0 : T : duration-T;

% dynamics settings
q_rcd = 2e-5;
q_amp = 0.1;
q_range = 0;
q_alpha = 0;
Q = blkdiag([T^3/3, T^2/2; T^2/2, T]*q_rcd, T*q_amp, T*q_range, T*q_alpha);

% initial state settings
init_rcd = -5;
init_rcd_rate = 0;
init_amp = CN02amp(40);
init_range = 300;
init_alpha = 0.6;

% transition matrix
F = [1 T 0 0 0; ...
     0 1 0 0 0; ...
     0 0 1 0 0; ...
     0 0 0 1 0; ...
     0 0 0 0 1];

% state ground truth
while true
    x_gt = RandomWalk([init_rcd; init_rcd_rate; init_amp; init_range; init_alpha], F, Q, length(time));
    if all(x_gt(3:5, :), 'all')
        break;
    end
end

% correlator offsets (chip)
% offsets = [-1.5 : 0.3 : -0.6, -0.5:0.1:0.8, 0.9 : 0.3 : 3];
offsets = -1 : 0.2 : 2;

% measurement noise covariance marix
R = 1/(4*T) * R_BPSK(offsets - offsets.');


%% correlation values generation

% % if try to reload from existing y signal to save some time
% try_reload = 1;
% 
% try 
%     tmp = load(sprintf("y_signal_%03d_%02d_%03d.mat", init_range, init_alpha*100, duration));
%     if try_reload && tmp.init_range == init_range && tmp.init_alpha == init_alpha && tmp.duration == duration 
%         y_signal = tmp.y_signal;
%     else
%         error("");
%     end
% catch
%     y_signal = x_gt(3,:) .* R_scatter(x_gt(1,:)/lc - offsets.', x_gt(4,:)/lc, x_gt(5,:));
%     save(sprintf("y_signal_%03d_%02d_%03d.mat", init_range, init_alpha*100, duration), "y_signal", "init_range", "init_alpha", "duration");
% end

Z_LUT_struct = load("Z_LUT.mat");
Z_LUT = Z_LUT_struct.Z_LUT;
delay_error_grid = Z_LUT_struct.delay_error_grid;
range_grid = Z_LUT_struct.range_grid;
alpha_grid = Z_LUT_struct.alpha_grid;

[range_mesh, delay_error_mesh, alpha_mesh] = meshgrid(range_grid, delay_error_grid, alpha_grid);

x_1345_lim = [[-1,1]*lc; ...
              0, Inf; ...
              range_grid([1,end])*lc; ...
              alpha_grid([1,end])];

y_signal = nan(length(offsets), length(time));

for ii = 1:length(offsets)
    y_signal(ii, :) = x_gt(3,:) .* interp3(range_mesh, delay_error_mesh, alpha_mesh, Z_LUT, ...
        x_gt(4,:)/lc, x_gt(1,:)/lc - offsets(ii), x_gt(5,:));
end

y_noise = mvnrnd(zeros(size(offsets)), R, length(time)).';
y = y_signal + y_noise;


figure; grid on; hold on;
plot(offsets, y(:, 1:5/T:end), '.-');
legend(num2str((0:5:duration-0.1).', '%02d') + " sec");

%% init settings for all filters

init_x_prior = [0; 0; CN02amp(30); 300; 1];
init_P_prior = diag([20, 0.1, 10, 60, 0.3]);

%% SIR

% filter forward, backward, forward ...
filtering_times = 1;

% log the results
for ii = 1 : filtering_times
    results.SIR(ii).x_post = nan(5, length(time));
    results.SIR(ii).P_post = nan(5, length(time));
end

% particle number
particle_num = 1e6;

h = waitbar(0, "SIR Processing ...");
for ii = 1 : filtering_times
    for jj = 1 : length(time)
        if jj == 1
            if ii == 1
                % if it's the first epoch in the first filtering
                % initialize the prior
                x_particle_prior = repmat(init_x_prior, 1, particle_num);
            else
                % if it's the first epoch but not first filtering
                % take the post estimate from last epoch of last round of filtering
                x_particle_prior = x_particle_post;
            end
        else
            % propagate the state to next epoch w/o dynamics
            x_particle_prior = F * x_particle_post;
            % add dynamics
            valid_indices = zeros(1, particle_num);
            x_particle_dynamics = nan(5, particle_num);
            while true
                % generate dynamics
                x_particle_dynamics(:, ~valid_indices) = mvnrnd(zeros(1,5), Q, sum(~valid_indices)).';
                % make sure that for all particles, the 1, 3, 4, 5 entries are valid
                % otherwise, regenerate dynamics for those particle that are not
                valid_indices = all(x_1345_lim(:, 1) < (x_particle_dynamics([1,3,4,5], :) + x_particle_prior([1,3,4,5], :)) ...
                    & (x_particle_dynamics([1,3,4,5], :) + x_particle_prior([1,3,4,5], :)) < x_1345_lim(:, 2));
                if all(valid_indices)
                    break;
                end
            end
            % prior w/ dynamics
            x_particle_prior = x_particle_prior + x_particle_dynamics;
        end
        % re-calculate weight
        y_particle_prior = nan(length(offsets), particle_num);
        % predict y
        for kk = 1 : length(offsets)
            y_particle_prior(kk, :) = x_particle_prior(3,:) .* interp3(range_mesh, delay_error_mesh, alpha_mesh, Z_LUT, ...
                x_particle_prior(4,:)/lc, x_particle_prior(1,:)/lc - offsets(kk), x_particle_prior(5,:));
        end
        % calculate weight based on difference between measured and predict y
        w_particle = mvnpdf((y(:, jj) - y_particle_prior).', zeros(1, length(offsets)), R).';
        % normalize weight
        w_particle = w_particle / sum(w_particle);

        % log the MMSE posterior x and P
        results.SIR(ii).x_post(:, jj) = x_particle_prior * w_particle.';
        results.SIR(ii).P_post(:, jj) = (x_particle_prior - results.SIR(ii).x_post(:, jj)).^2 * w_particle.';

        % resampling
        x_particle_post = sampling(x_particle_prior, w_particle);
        
        if mod(jj, 100) == 0
            waitbar(jj/length(time), h);
        end
    end

end
