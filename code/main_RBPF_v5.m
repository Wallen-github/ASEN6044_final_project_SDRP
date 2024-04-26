%--------------------------------------------------------------------------
% Name: ASEN6044 Project
%
% Desc: This code is part of ASEN 6044 final project, mainly focusing on
% the Rao-Blackwellized particle filter
%
% Author: Hai-Shuo Wang
% Affiliation: Univercity of Colorado Boulder, CSML
% Time: 04/24/2024
% Version 5.0:
%--------------------------------------------------------------------------
clc;
clear;
close all;
format LONG;
addpath(genpath("./"))

%% parameters setup
% speed of light
param.c = 299792458;
% chipping rate (Hz)
param.fc = 1.023e6;
% chiplength (m)
param.lc = param.c / param.fc;

% coherent integration time
param.T = 20e-3;
% simulation duration
param.duration = 1;
% timeline
param.time = 0 : param.T : param.duration-param.T;

% dynamics setup
param.dynamics.q_rcd = 2e-5;
param.dynamics.q_amp = 0.1;
param.dynamics.q_range = 0;
param.dynamics.q_alpha = 0;
param.dynamics.Q = blkdiag([param.T^3/3, param.T^2/2; param.T^2/2, param.T]...
    *param.dynamics.q_rcd, param.T*param.dynamics.q_amp, ...
    param.T*param.dynamics.q_range, param.T*param.dynamics.q_alpha);

% transition matrix
param.dynamics.F = [1 param.T 0 0 0; ...
     0 1 0 0 0; ...
     0 0 1 0 0; ...
     0 0 0 1 0; ...
     0 0 0 0 1];

Z_LUT_struct = load("Z_LUT.mat");
range_grid = Z_LUT_struct.range_grid;
alpha_grid = Z_LUT_struct.alpha_grid;
param.dynamics.x_1345_lim = [[-1,1]*param.lc; ...
                  0, Inf; ...
                  range_grid([1,end])*param.lc; ...
                  alpha_grid([1,end])];

% correlator offsets (chip)
param.measure.offsets = -1 : 0.2 : 2;
% measurement noise covariance marix
param.measure.R = 1/(4*param.T) * R_BPSK(param.measure.offsets - param.measure.offsets.');

%% initial state setup
init_rcd = -5;
init_rcd_rate = 0;
init_amp = CN02amp(40);
init_range = 300;
init_alpha = 1.2;
state_init = [init_rcd; init_rcd_rate; init_amp; init_range; init_alpha];
covar_init = diag([20, 0.1, 10, 60, 0.3]);

% Ground truth state and measurement

% Get the ground truth state
x_gt = state_ground_truth(param, state_init);

% Get the ground truth measurement
y_gt = measure_ground_truth(param, Z_LUT_struct, x_gt);

%% Rao-Blackwellized Particle Filtering

% PF setup
num_particle = 100;
num_measurment = size(y_gt,2);
resample_percentage = 0.3;

x_particle_post = sampling_v2([linspace(-50,50,20); ...
                             linspace(3, -3, 20); ...
                             CN02amp([linspace(20,50,20)]); ...
                             linspace(15,500, 20); ...
                             linspace(0.3, 1.9, 20)], ...
                             ones(1,20)/20, num_particle);

state_post = zeros(length(state_init),num_measurment);
sigma = zeros(length(state_init),num_measurment);
state_post(:,1) = state_init;
covar_post_pf = zeros(length(state_init),length(state_init),num_measurment);
covar_post_pf(:,:,1) = covar_init;
covar_post_ukf = repmat(covar_init(1:2,1:2),1,1,num_particle);
% covar_post_ukf(:,:,1) = covar_init(1:2,1:2);
meaure_post = nan(size(y_gt,1),num_measurment-1);
sigma(:,1) = sqrt(diag(covar_init));

dim = length(state_init);

t_start = clock;
for i=2:num_measurment
    if mod(i,50)==0
        t_current = clock;
        t_elapsed = etime(t_current,t_start);
        t_remain = t_elapsed/i * (num_measurment - i);
        fprintf("loop=%i/%i, Elapsed time: %4.3f s, remaining time: %4.3f s\n",i,num_measurment,t_elapsed, t_remain);
    end

    % state update for pf
    x_particle_prior = state_update(param, x_particle_post,num_particle);
    covar_prior_ukf = state_covar_update(param,covar_post_ukf,2,num_particle);
    y_particle_prior = measure_ground_truth(param, Z_LUT_struct, x_particle_prior);
    
    parfor k=1:num_particle
        [x_particle_post_ukf(:,k),covar_post_ukf(:,:,k)] = UKF_measure_update(x_particle_prior(:,k)...
                ,covar_prior_ukf(:,:,k), y_particle_prior(:,k), y_gt(:,i), param, Z_LUT_struct, 1);
    end

    % weight update
    err = (y_gt(:, i) - y_particle_prior).';
    weight = mvnpdf(err, zeros(1, length(param.measure.offsets)), param.measure.R).';
    % normalize weight
    weight = weight / sum(weight);

    % Posterior state and varaince
    state_post(:,i) = x_particle_post_ukf * weight.';
    covar = zeros(5,5);
    covar_ukf = zeros(2,2);
    for j=1:num_particle
        covar = covar + (x_particle_post_ukf(:,j) - state_post(:,i))*...
            (x_particle_post_ukf(:,j) - state_post(:,i))'*weight(j);
        covar_ukf = covar_ukf + covar_post_ukf(:,:,j)*weight(j);
    end
    covar_post_pf(:,:,i) = covar;
    sigma(:,i) = sqrt(diag(covar));
    
%     sigma(1:2,i) = sqrt(diag(covar_ukf));
%     sigma(3:5,i) = sqrt(diag(covar(3:5,3:5)));

    % compute Ness
    Ness = 1/sum(weight.^2);    
    Nt = resample_percentage*num_particle;
    if Ness<Nt
        % resampling
%         fprintf("resampling at ith step: %i \n", i);
        x_particle_post(3:5,:) = sampling_v2(x_particle_prior(3:5,:), weight,num_particle);
        weight = repmat(1/num_particle,1,num_particle);
    else
        x_particle_post = x_particle_prior;
    end

end

figure
for i=1:size(x_gt,1)
    subplot(5,1,i)
    hold on;
    fill([param.time,fliplr(param.time)],...
        [state_post(i,:)-2*sigma(i,:),fliplr(state_post(i,:)+2*sigma(i,:))],...
        [0.7,0.7,0.7],FaceAlpha=0.3,EdgeColor=[0.7,0.7,0.7],DisplayName="2-\sigma");
    plot(param.time,state_post(i,:),Color='red',DisplayName='RBPF')
    plot(param.time, x_gt(i,:),Color='blue',DisplayName='truth')
    if i==1 
        title("RBPF State"); 
        legend; 
    end
    grid on
%     xlim([0.5 5])
    ylabel(['x(' num2str(i) ')'])
%     set(gca,Fontsize=20,FontWeight='bold')
end

figure
for i=1:size(x_gt,1)
    subplot(5,1,i)
    plot(param.time,x_gt(i,:) - state_post(i,:))
    grid on
    ylabel(['x(' num2str(i) ') error'])
    if i==1 
        title("RBPF Error"); 
    end
    xlabel("time (s)")
%     set(gca,Fontsize=20,FontWeight='bold')
%     xlim([0.5 10])
end


%% functions

function [x_post,Px_post] = UKF_measure_update(x_prior,Px_prior, y_prior, y_truth, param, Z_LUT_struct,num_particle)

    dim_y = length(param.measure.R);
    dim = 2;%size(x_prior,1);
    Pyy = zeros(dim_y,dim_y);
    Pxy = zeros(dim,dim_y);
    ka = 0;
    alpha = 1e-2;
    beta = 2;
    lam = alpha^2*(dim+ka) - dim;

    % generate 2*dim +1 sigma points
    for i=1:num_particle
        Sk = chol(Px_prior(:,:,i));
        chi(:,1) = x_prior(1:2,i);
        y_chi(:,1) = measure_ground_truth(param, Z_LUT_struct, x_prior(:,i));
        wm = 0.5/(dim+lam)*ones(2*dim+1,1);
        wc = wm;
        wm(1) = lam/(dim+lam);
        wc(1) = lam/(dim+lam) + 1 - alpha^2 + beta;
        for j=1:dim
            chi(:,j+1) = x_prior(1:2,i) + sqrt(dim+lam)*Sk(j,:)';
            chi(:,j+dim+1) = x_prior(1:2,i) - sqrt(dim+lam)*Sk(j,:)';
    
            % propogate 2*dim +1 sigma points
            y_chi(:,j+1) = measure_ground_truth(param, Z_LUT_struct, [chi(:,j+1);x_prior(3:5,i)]);
            y_chi(:,j+dim+1) = measure_ground_truth(param, Z_LUT_struct, [chi(:,j+dim+1);x_prior(3:5,i)]);
        end
    
        % get predicted measurements mean and covariance
        y_post = y_chi*wm;
        
        for k=1*2*dim+1
    
            Pyy = Pyy + wc(k)*(y_chi(:,k) - y_truth)*(y_chi(:,k) - y_truth)' + param.measure.R;
            Pxy = Pxy + wc(k)*(chi(:,k) - x_prior(1:2,i))*(y_chi(:,k) - y_truth)';
    
        end
        kalman_gain = Pxy/Pyy;
    
        x_post12 = x_prior(1:2,i) + kalman_gain*(y_prior(:,i) - y_post);
        x_post(:,i) = [x_post12;x_prior(3:5,i)];
        Px_post(:,:,i) = Px_prior(:,:,i) - kalman_gain*Pyy*kalman_gain';
    end

end

function x_particle_k1 = state_update(param, x_particle_k,num_particle)

    % propagate the state to next epoch w/o dynamics
    x_particle_k1 = param.dynamics.F * x_particle_k;
    % add dynamics
    valid_indices = zeros(1, num_particle);
    x_particle_dynamics = nan(5, num_particle);
    while true
        % generate dynamics
        x_particle_dynamics(:, ~valid_indices) = mvnrnd(zeros(1,5), param.dynamics.Q, sum(~valid_indices)).';
        % make sure that for all particles, the 1, 3, 4, 5 entries are valid
        % otherwise, regenerate dynamics for those particle that are not
        valid_indices = all(param.dynamics.x_1345_lim(:, 1) < (x_particle_dynamics([1,3,4,5], :) + x_particle_k1([1,3,4,5], :)) ...
            & (x_particle_dynamics([1,3,4,5], :) + x_particle_k1([1,3,4,5], :)) < param.dynamics.x_1345_lim(:, 2));
        if all(valid_indices)
            break;
        end
    end
    % prior w/ dynamics
    x_particle_k1 = x_particle_k1 + x_particle_dynamics;

end

function P_particle_k1 = state_covar_update(param,P_particle_k,dim_easy,num_particle)

    P_particle_k1 = nan(dim_easy,dim_easy,num_particle);
    for i=1:num_particle
        P_particle_k1(:,:,i) = param.dynamics.F(1:dim_easy,1:dim_easy) * P_particle_k(:,:,i) * param.dynamics.F(1:dim_easy,1:dim_easy)' + param.dynamics.Q(1:dim_easy,1:dim_easy);
    end
end

function [x_gt] = state_ground_truth(param, state_init)
    
    % state ground truth
    while true
        x_gt = RandomWalk(state_init, param.dynamics.F, param.dynamics.Q, length(param.time));
        if all(x_gt(3:5, :), 'all')
            break;
        end
    end
end

function [y_gt] = measure_ground_truth(param, Z_LUT_struct, x_gt)

    Z_LUT = Z_LUT_struct.Z_LUT;
    delay_error_grid = Z_LUT_struct.delay_error_grid;
    range_grid = Z_LUT_struct.range_grid;
    alpha_grid = Z_LUT_struct.alpha_grid;
    
    [range_mesh, delay_error_mesh, alpha_mesh] = meshgrid(range_grid, delay_error_grid, alpha_grid);
    
    
    y_signal = nan(length(param.measure.offsets), size(x_gt,2));
    
    for ii = 1:length(param.measure.offsets)
        y_signal(ii, :) = x_gt(3,:) .* interp3(range_mesh, delay_error_mesh, alpha_mesh, Z_LUT, ...
            x_gt(4,:)/param.lc, x_gt(1,:)/param.lc - param.measure.offsets(ii), x_gt(5,:));
    end
    
    y_noise = mvnrnd(zeros(size(param.measure.offsets)), param.measure.R, size(x_gt,2)).';
    y_gt = y_signal + y_noise;

end


