%--------------------------------------------------------------------------
% Name: ASEN6044 Project
%
% Desc: This code is part of ASEN 6044 final project, mainly focusing on
% the Rao-Blackwellized particle filter
%
% Author: Hai-Shuo Wang
% Affiliation: Univercity of Colorado Boulder, CSML
% Time: 04/23/2024
% Version 1.0:
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
[x_gt,F,Q] = state_ground_truth(state_init);

offsets = -1 : 0.2 : 2;
[y_gt,R] = measurement(x_gt, offsets);

% PF setup
num_particle = 100;
weight = 1/num_particle;
x_particle_post = repmat(state_init, 1, num_particle);
state_post(:,1) = state_init;
covar_post(:,:,1) = covar_init;

h = timebar('Progress','FastSLAM Simulation');  % Start timebar
num_measurment = size(y_gt,2);
for i=2:num_measurment
    timebar(h, i/num_measurment)

    % Kalman state update
    state_prior = F * state_post(:,i-1) + mvnrnd(zeros(5,1), Q, 1)';
    covar_prior = F * covar_post(:,:,i-1) * F' + Q;
    
    % state update
    for k=1:num_particle
        x_particle_prior(:,k) = F * x_particle_post(:,k) + mvnrnd(zeros(5,1), Q, 1)';
    end
    x_particle_prior(1:2,:) = repmat(state_prior(1:2,1), 1, num_particle);

    % measurement update
    y_particle_prior = nan(length(offsets), num_particle);
    for j=1:num_particle
        y_particle_prior(:,j) = measurement(x_particle_prior(:,j),offsets);
    end

    % weight update
    err = (y_gt(:, i) - y_particle_prior).';
    weight = mvnpdf(err, zeros(1, length(offsets)), R).';
    % normalize weight
    weight = weight / sum(weight);

    % Posterior state and varaince
    state_post(1:2,i) = state_prior(1:2,1);
    covar_post(1:2,1:2,i) = covar_prior(1:2,1:2);
    state_post(3:5,i) = x_particle_prior(3:5,:) * weight.';
    covar = zeros(3,3);
    for j=1:num_particle
        covar = covar + (x_particle_prior(3:5,j) - state_post(3:5,i))*(x_particle_prior(3:5,j) - state_post(3:5,i))'*weight(j);
    end
    covar_post(3:5,3:5,i) = covar;

    % compute Ness
    Ness = 1/sum(weight.^2);
    resample_percentage = 0.3;    
    Nt = resample_percentage*num_particle;
    if Ness<Nt
        disp("resampling")
        % resampling
        x_particle_post = sampling(x_particle_prior, weight);
        x_particle_post(1:2,:) = repmat(state_post(1:2,1), 1, num_particle);
    else
        x_particle_post = x_particle_prior;
    end

    
end



%% functions

function [x_gt,F,Q] = state_ground_truth(state_init)

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

function [y,R] = measurement(x_gt,offsets)
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