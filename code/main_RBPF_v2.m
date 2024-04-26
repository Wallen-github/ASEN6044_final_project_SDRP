%--------------------------------------------------------------------------
% Name: ASEN6044 Project
%
% Desc: This code is part of ASEN 6044 final project, mainly focusing on
% the Rao-Blackwellized particle filter
%
% Author: Hai-Shuo Wang
% Affiliation: Univercity of Colorado Boulder, CSML
% Time: 04/24/2024
% Version 2.0:
%--------------------------------------------------------------------------
clc;
clear;
close all;
format LONG;
addpath(genpath("./"))

% initial state settings
init_rcd = -5;
init_rcd_rate = 0;
init_amp = CN02amp(40);
init_range = 300;
init_alpha = 0.6;
state_init = [init_rcd; init_rcd_rate; init_amp; init_range; init_alpha];
covar_init = diag([20, 0.1, 10, 60, 0.3]);

% Ground truth state and measurement
[x_gt,F,Q,time] = state_ground_truth(state_init);

offsets = -1 : 0.2 : 2;
[y_gt,R,range_mesh, delay_error_mesh, alpha_mesh, Z_LUT_struct, lc] = measurement(x_gt, offsets);
Z_LUT = Z_LUT_struct.Z_LUT;

% PF setup
num_particle = 100;
num_measurment = size(y_gt,2);

% x_particle_post = repmat(state_init, 1, num_particle);
x_particle_post = sampling_v2([linspace(-50,50,20); ...
                             linspace(3, -3, 20); ...
                             CN02amp([linspace(20,50,20)]); ...
                             linspace(15,500, 20); ...
                             linspace(0.3, 1.9, 20)], ...
                             ones(1,20)/20, num_particle);
state_post = zeros(length(state_init),num_measurment);
covar_post = zeros(length(state_init),length(state_init),num_measurment);
state_post(:,1) = state_init;
covar_post(:,:,1) = covar_init;

h = timebar('Progress','RBPF Simulation');  % Start timebar

for i=2:num_measurment
    if mod(i,50)==0
%         timebar(h, i/num_measurment)
        disp(i/num_measurment)
    end
    fprintf("loop=%d\n",i/num_measurment);

    % Kalman state update
    x_noise = addnoise(state_post(:,i-1),Q,lc,Z_LUT_struct);
    state_prior = F * state_post(:,i-1) + x_noise;
    covar_prior = F * covar_post(:,:,i-1) * F' + Q;
    
    % state update and measurement update
    y_particle_prior = nan(length(offsets), num_particle);
    parfor k=1:num_particle
        x_noise = addnoise(x_particle_post(:,k),Q,lc,Z_LUT_struct);
        x_particle_prior(:,k) = F * x_particle_post(:,k) + x_noise;
        x_particle_prior(1:2,k) = state_prior(1:2,1);
%         y_particle_prior(:,k) = measurement(x_particle_prior(:,k),offsets);
        for kk = 1 : length(offsets)
            y_particle_prior(kk, :) = x_particle_prior(3,k) .* interp3(range_mesh, delay_error_mesh, alpha_mesh, Z_LUT, ...
                x_particle_prior(4,k)/lc, x_particle_prior(1,k)/lc - offsets(kk), x_particle_prior(5,k));
        end
        
        err = y_gt(:, i) - y_particle_prior(:,k);
        weight(k) = mvnpdf(err, zeros(length(offsets),1), R).';
        state_post(3:5,i) = state_post(3:5,i) + x_particle_prior(3:5,k) * weight(k);
        covar_post(3:5,3:5,i) = covar_post(3:5,3:5,i) ...
            + (x_particle_prior(3:5,k) - state_post(3:5,i))*(x_particle_prior(3:5,k) - state_post(3:5,i))'*weight(k);
    end

    % Posterior state and varaince
    state_post(1:2,i) = state_prior(1:2,1);
    covar_post(1:2,1:2,i) = covar_prior(1:2,1:2);

    state_post(3:5,i) = state_post(3:5,i)/sum(weight);
    covar_post(3:5,3:5,i) = covar_post(3:5,3:5,i)/sum(weight);

    % compute Ness
    Ness = 1/sum(weight.^2);
    resample_percentage = 0.3;    
    Nt = resample_percentage*num_particle;
    if Ness<Nt
        disp("resampling")
        % resampling
        x_particle_post = sampling_v2(x_particle_prior, weight,num_particle);
        x_particle_post(1:2,:) = repmat(state_post(1:2,1), 1, num_particle);
    else
        x_particle_post = x_particle_prior;
    end

end

for i=1:size(covar_post,3)
    sigma(:,i) = sqrt(diag(covar_post(:,:,i)));
end

figure
for i=1:size(x_gt,1)
    subplot(5,1,i)
    grid on
    hold on
    plot(time, x_gt(i,:),Color='blue')
    plot(time,state_post(i,:),Color='red')
    plot(time,state_post(i,:)+2*sigma(i,:),Color = 'green')
    plot(time,state_post(i,:)-2*sigma(i,:),Color = 'green')
end




%% functions

function x_particle_dynamics = addnoise(x_particle_prior,Q,lc,Z_LUT_struct)
    range_grid = Z_LUT_struct.range_grid;
    alpha_grid = Z_LUT_struct.alpha_grid;
    x_1345_lim = [[-1,1]*lc; ...
                  0, Inf; ...
                  range_grid([1,end])*lc; ...
                  alpha_grid([1,end])];
    x_particle_dynamics = nan(5, 1);
    while true
        % generate dynamics
        x_particle_dynamics = mvnrnd(zeros(1,5), Q).';
        % make sure that for all particles, the 1, 3, 4, 5 entries are valid
        % otherwise, regenerate dynamics for those particle that are not
        valid = all(x_1345_lim(:, 1) < (x_particle_dynamics([1,3,4,5]) + x_particle_prior([1,3,4,5])) ...
            & (x_particle_dynamics([1,3,4,5]) + x_particle_prior([1,3,4,5])) < x_1345_lim(:, 2));
        if valid
            break;
        end
    end

end

function [x_gt,F,Q,time] = state_ground_truth(state_init)

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
        x_gt = RandomWalk(state_init, F, Q, length(time));
        if all(x_gt(3:5, :), 'all')
            break;
        end
    end
end

function [y,R,range_mesh, delay_error_mesh, alpha_mesh, Z_LUT_struct, lc] = measurement(x_gt,offsets)
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

    % correlator offsets (chip)
    % offsets = [-1.5 : 0.3 : -0.6, -0.5:0.1:0.8, 0.9 : 0.3 : 3];
%     offsets = -1 : 0.2 : 2;
    
    % measurement noise covariance marix
    R = 1/(4*T) * R_BPSK(offsets - offsets.');
    
    
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
    
    y_signal = nan(length(offsets), size(x_gt,2));
    
    for ii = 1:length(offsets)
        y_signal(ii, :) = x_gt(3,:) .* interp3(range_mesh, delay_error_mesh, alpha_mesh, Z_LUT, ...
            x_gt(4,:)/lc, x_gt(1,:)/lc - offsets(ii), x_gt(5,:));
    end
    
    y_noise = mvnrnd(zeros(size(offsets)), R, size(x_gt,2)).';
    y = y_signal + y_noise;

end