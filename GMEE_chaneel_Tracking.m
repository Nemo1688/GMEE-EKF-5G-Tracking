%% 5G通道估計 - GMEE-EKF vs LS 比較
% 應用於時變通道追蹤

clear; close all; clc;

%% 1. 系統參數設定
fprintf('=== 5G通道估計系統初始化 ===\n');

% 5G系統參數
fc = 28e9;              % 載波頻率 (28 GHz)
c = 3e8;                % 光速
lambda = c / fc;        % 波長

% 通道參數
Npath = 4;              % 多徑數量
Nsym = 100;             % OFDM符號數
SNR_dB = 15;            % 信噪比 (dB)

% 移動性參數
v = 120 / 3.6;          % 速度 (120 km/h)
fd = v * fc / c;        % 最大都卜勒頻移
Ts = 1e-5;              % 符號週期

fprintf('載波頻率: %.2f GHz\n', fc/1e9);
fprintf('移動速度: %.1f km/h\n', v*3.6);
fprintf('都卜勒頻移: %.2f Hz\n', fd);
fprintf('多徑數量: %d\n', Npath);

%% 2. 生成時變通道
fprintf('\n=== 生成時變5G通道 ===\n');

channel_taps = generate_5g_channel(Npath, Nsym, fd, Ts);
true_channel = channel_taps;

% 添加雜訊
SNR_linear = 10^(SNR_dB/10);
noise_power = 1 / SNR_linear;
noisy_channel = channel_taps + sqrt(noise_power/2) * (randn(size(channel_taps)) + 1j*randn(size(channel_taps)));

fprintf('通道生成完成\n');

%% 3. GMEE-EKF 追蹤
fprintf('\n=== GMEE-EKF 追蹤 ===\n');

% 初始化
n_state = Npath * 4;    % [h_real; h_imag; h_dot_real; h_dot_imag]
n_meas = Npath * 2;

x_est = zeros(n_state, 1);
x_est(1:Npath) = real(noisy_channel(:, 1));
x_est(Npath+1:2*Npath) = imag(noisy_channel(:, 1));

P_est = eye(n_state) * 0.1;

q_factor = 2 * pi * fd * Ts;
Q = eye(n_state) * (q_factor^2);
R = eye(n_meas) * noise_power;

Phi = build_state_transition(Npath, fd, Ts);

gmee_channel = zeros(Npath, Nsym);
gmee_mse = zeros(Nsym, 1);

for t = 1:Nsym
    z = [real(noisy_channel(:, t)); imag(noisy_channel(:, t))];
    [x_est, P_est] = GMEE_EKF_5G(x_est, P_est, z, R, Phi, Q, Npath);
    
    h_est = x_est(1:Npath) + 1j * x_est(Npath+1:2*Npath);
    gmee_channel(:, t) = h_est;
    gmee_mse(t) = mean(abs(h_est - true_channel(:, t)).^2);
    
    if mod(t, 20) == 0
        fprintf('GMEE進度: %d/%d\n', t, Nsym);
    end
end

fprintf('GMEE追蹤完成\n');

%% 4. LS 估計
fprintf('\n=== LS 估計 ===\n');

ls_channel = noisy_channel;  % LS直接使用量測值
ls_mse = zeros(Nsym, 1);

for t = 1:Nsym
    ls_mse(t) = mean(abs(ls_channel(:, t) - true_channel(:, t)).^2);
end

fprintf('LS估計完成\n');

%% 5. 性能比較
fprintf('\n=== 性能比較 ===\n');

fprintf('GMEE-EKF 平均MSE: %.6f\n', mean(gmee_mse));
fprintf('LS 平均MSE: %.6f\n', mean(ls_mse));
improvement = (mean(ls_mse) - mean(gmee_mse)) / mean(ls_mse) * 100;
fprintf('GMEE相對改善: %.2f%%\n', improvement);

%% 6. 視覺化
figure('Position', [100, 100, 1400, 800]);

% 1. 通道幅度追蹤比較
subplot(2,3,1);
plot(1:Nsym, abs(true_channel(1,:)), 'k-', 'LineWidth', 2);
hold on;
plot(1:Nsym, abs(gmee_channel(1,:)), 'r-', 'LineWidth', 1.5);
plot(1:Nsym, abs(ls_channel(1,:)), 'g--', 'LineWidth', 1.5);
grid on;
xlabel('OFDM符號索引');
ylabel('通道增益');
title('路徑1: 通道幅度追蹤');
legend('真實通道', 'GMEE-EKF', 'LS', 'Location', 'best');

% 2. 通道相位追蹤比較
subplot(2,3,2);
plot(1:Nsym, angle(true_channel(1,:)), 'k-', 'LineWidth', 2);
hold on;
plot(1:Nsym, angle(gmee_channel(1,:)), 'r-', 'LineWidth', 1.5);
plot(1:Nsym, angle(ls_channel(1,:)), 'g--', 'LineWidth', 1.5);
grid on;
xlabel('OFDM符號索引');
ylabel('相位 (弧度)');
title('路徑1: 通道相位追蹤');
legend('真實通道', 'GMEE-EKF', 'LS', 'Location', 'best');

% 3. MSE比較
subplot(2,3,3);
semilogy(1:Nsym, gmee_mse, 'r-', 'LineWidth', 2);
hold on;
semilogy(1:Nsym, ls_mse, 'g--', 'LineWidth', 2);
grid on;
xlabel('OFDM符號索引');
ylabel('MSE');
title('均方誤差比較');
legend('GMEE-EKF', 'LS', 'Location', 'best');

% 4. 所有路徑 - GMEE
subplot(2,3,4);
for p = 1:Npath
    plot(1:Nsym, abs(gmee_channel(p,:)), 'LineWidth', 1.5);
    hold on;
end
grid on;
xlabel('OFDM符號索引');
ylabel('通道增益');
title('GMEE-EKF: 所有路徑');
legend(arrayfun(@(x) sprintf('路徑%d', x), 1:Npath, 'UniformOutput', false));

% 5. 所有路徑 - LS
subplot(2,3,5);
for p = 1:Npath
    plot(1:Nsym, abs(ls_channel(p,:)), 'LineWidth', 1.5);
    hold on;
end
grid on;
xlabel('OFDM符號索引');
ylabel('通道增益');
title('LS: 所有路徑');
legend(arrayfun(@(x) sprintf('路徑%d', x), 1:Npath, 'UniformOutput', false));

% 6. 性能改善百分比
subplot(2,3,6);
improvement_vec = (ls_mse - gmee_mse) ./ ls_mse * 100;
plot(1:Nsym, improvement_vec, 'g-', 'LineWidth', 2);
hold on;
yline(mean(improvement_vec), 'r--', 'LineWidth', 1.5);
grid on;
xlabel('OFDM符號索引');
ylabel('改善百分比 (%)');
title(sprintf('GMEE相對LS改善 (平均: %.2f%%)', mean(improvement_vec)));

sgtitle('5G通道估計: GMEE-EKF vs LS', 'FontSize', 14, 'FontWeight', 'bold');

%% ============ 輔助函數 ============

function channel = generate_5g_channel(Npath, Nsym, fd, Ts)
    % =======================================================
    % Clarke's / Jakes channel model (Rayleigh fading)
    % 模擬 5G 多徑快衰落通道
    %
    % 輸入:
    %   Npath   - 多徑數量
    %   Nsym    - 符號數
    %   fd      - 最大都卜勒頻移 (Hz)
    %   Ts      - 符號週期 (s)
    %
    % 輸出:
    %   channel - [Npath x Nsym] 複數通道增益
    %
    % =======================================================
    
    % 設定 sinusoids 數量 (Jakes 建議 >= 8，越多越準確)
    N0 = 16;  
    
    % 初始化通道矩陣
    channel = zeros(Npath, Nsym);
    
    % 每條 path
    for p = 1:Npath
        % 隨機初始相位
        phi0 = rand * 2 * pi;  
        % 路徑增益 (隨機 or 遞減)
        gain = exp(-0.5 * (p-1)); 
        
        % Clarke 模型: 每條路徑由 N0 個正弦波組合
        n = 1:N0;
        beta_n = (pi * n) / (N0 + 1);    % 均勻分布角度
        theta_n = rand(1, N0) * 2*pi;    % 隨機相位偏移
        
        for t = 1:Nsym
            time = (t-1) * Ts;
            
            % Jakes 模型公式：sum of sinusoids
            re_part = sum(cos(2*pi*fd*time*cos(beta_n) + theta_n));
            im_part = sum(sin(2*pi*fd*time*cos(beta_n) + theta_n));
            
            h = (re_part + 1j*im_part) / sqrt(2*N0);
            
            % 乘上路徑增益 & 初始相位
            channel(p, t) = gain * h * exp(1j*phi0);
        end
    end
end


function Phi = build_state_transition(Npath, fd, Ts)
    % 建立狀態轉移矩陣
    n = Npath * 4;
    Phi = eye(n);
    
    for i = 1:Npath
        Phi(i, 2*Npath+i) = Ts;
        Phi(Npath+i, 3*Npath+i) = Ts;
    end
    
    decay = exp(-0.1 * 2 * pi * fd * Ts);
    for i = 1:Npath
        Phi(2*Npath+i, 2*Npath+i) = decay;
        Phi(3*Npath+i, 3*Npath+i) = decay;
    end
end

function [x_est, P_est] = GMEE_EKF_5G(x_est, P_est, z, R, Phi, Q, Npath)
    % GMEE-EKF for 5G通道估計
    
    n = length(x_est);
    m = length(z);
    
    % 預測
    x_pred = Phi * x_est;
    P_pred = Phi * P_est * Phi' + Q;
    
    % 量測矩陣
    H = [eye(Npath), zeros(Npath, n-Npath);
         zeros(Npath, Npath), eye(Npath), zeros(Npath, n-2*Npath)];
    
    z_pred = H * x_pred;
    
    % 構建聯合協方差
    B_all = [P_pred, zeros(n, m); zeros(m, n), R];
    B_all = (B_all + B_all') / 2 + 1e-9 * eye(size(B_all));
    
    try
        B = chol(B_all, 'lower');
    catch
        [V, D] = eig(B_all);
        D = max(D, 1e-9);
        B_all = V * D * V';
        B = chol(B_all, 'lower');
    end
    
    D = pinv(B) * [x_pred; z];
    x_iter = x_pred;
    
    % MEE參數
    alpha = 1.0;
    beta = 15.0;
    alpha_min = 0.5;
    alpha_max = 1.5;
    beta_min = 10.0;
    beta_max = 30.0;
    
    % 迭代優化
    max_iter = 15;
    tol = 1e-6;
    
    for iter = 1:max_iter
        x_old = x_iter;
        z_pred = H * x_iter;
        
        % 殘差
        W = pinv(B) * [x_iter; z_pred];
        e = D - W;
        error_norm = norm(e);
        
        % 自適應參數
        if error_norm < 5
            alpha = min(alpha_max, alpha + 0.05);
            beta = max(beta_min, beta - 0.5);
        elseif error_norm > 20
            alpha = max(alpha_min, alpha - 0.05);
            beta = min(beta_max, beta + 1.0);
        end
        
        % 計算權重
        L = n + m;
        psi = zeros(L, 1);
        gamma_val = gamma(1/alpha);
        
        for j = 1:L
            psi(j) = (alpha / (2 * beta * gamma_val)) * exp(-abs(e(j)/beta)^alpha);
            psi(j) = max(psi(j), 1e-12);
        end
        psi = psi / sum(psi);
        
        % 熵矩陣
        Psi = diag(psi);
        C = Psi - psi * psi';
        C = (C + C') / 2 + 1e-9 * eye(L);
        
        % 協方差分解
        B_n_inv = pinv(B(1:n, 1:n));
        B_m_inv = pinv(B(n+1:n+m, n+1:n+m));
        
        P_xx = B_n_inv' * C(1:n, 1:n) * B_n_inv;
        P_yx = B_n_inv' * C(1:n, n+1:n+m) * B_m_inv;
        R1 = B_m_inv' * C(n+1:n+m, n+1:n+m) * B_m_inv;
        
        % 卡爾曼增益
        K_mat = P_xx + H' * R1 * H + 1e-8 * eye(n);
        K = K_mat \ (P_yx + H' * R1);
        
        % 狀態更新
        innovation = z - z_pred;
        x_iter = x_pred + K * innovation;
        
        % 收斂檢查
        if norm(x_iter - x_old) / (norm(x_old) + 1e-10) < tol
            break;
        end
    end
    
    x_est = x_iter;
    
    % 協方差更新
    P_est = (eye(n) - K * H) * P_pred * (eye(n) - K * H)' + K * R * K';
    P_est = (P_est + P_est') / 2;
    [V, D] = eig(P_est);
    D = max(D, 1e-10);
    P_est = V * D * V';
end