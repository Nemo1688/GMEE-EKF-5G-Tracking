%% 5G通道估計 - GMEE-EKF vs LS 
% 應用於時變通道追蹤

clear; close all; clc;

%% 1. 系統參數設定
fprintf('=== 5G通道估計系統初始化 ===\n');

% 5G系統參數
fc = 28e9;              % 載波頻率 (28 GHz)
c = 3e8;                % 光速

% 通道參數
Npath = 4;              % 多徑數量
Nsym = 100;             % OFDM符號數
SNR_dB = 15;            % 信噪比 (dB)

% 移動性參數
v = 120 / 3.6;          % 速度 (120 km/h)
fd = v * fc / c;        % 最大都卜勒頻移
Ts = 1e-5;              % 符號週期

% 蒙地卡羅模擬次數
N_monte = 100;

fprintf('載波頻率: %.2f GHz\n', fc/1e9);
fprintf('移動速度: %.1f km/h\n', v*3.6);
fprintf('都卜勒頻移: %.2f Hz\n', fd);
fprintf('多徑數量: %d\n', Npath);
fprintf('蒙地卡羅次數: %d\n', N_monte);

%% 2. 初始化統計變數
gmee_mse_all = zeros(N_monte, Nsym);
ls_mse_all = zeros(N_monte, Nsym);
gmee_avg_mse = zeros(N_monte, 1);
ls_avg_mse = zeros(N_monte, 1);

n_state = Npath * 4;
n_meas = Npath * 2;
q_factor = 2 * pi * fd * Ts;
Q = eye(n_state) * (q_factor^2);
Phi = build_state_transition(Npath, fd, Ts);

%% 3. 蒙地卡羅模擬主循環
fprintf('\n=== 開始 %d 次蒙地卡羅模擬 ===\n', N_monte);
tic;

for mc = 1:N_monte
    % 生成時變通道
    true_channel = generate_5g_channel(Npath, Nsym, fd, Ts);
    
    % 添加雜訊
    SNR_linear = 10^(SNR_dB/10);
    noise_power = 1 / SNR_linear;
    noisy_channel = true_channel + sqrt(noise_power/2) * ...
        (randn(size(true_channel)) + 1j*randn(size(true_channel)));
    
    % GMEE-EKF 初始化
    x_est = zeros(n_state, 1);
    x_est(1:Npath) = real(noisy_channel(:, 1));
    x_est(Npath+1:2*Npath) = imag(noisy_channel(:, 1));
    P_est = eye(n_state) * 0.1;
    R = eye(n_meas) * noise_power;
    
    gmee_channel = zeros(Npath, Nsym);
    ls_channel = noisy_channel;  % LS直接使用量測值
    
    % 逐符號處理
    for t = 1:Nsym
        % GMEE-EKF 追蹤
        z = [real(noisy_channel(:, t)); imag(noisy_channel(:, t))];
        [x_est, P_est] = GMEE_EKF_5G(x_est, P_est, z, R, Phi, Q, Npath);
        h_est = x_est(1:Npath) + 1j * x_est(Npath+1:2*Npath);
        gmee_channel(:, t) = h_est;
        
        % 計算MSE
        gmee_mse_all(mc, t) = mean(abs(h_est - true_channel(:, t)).^2);
        ls_mse_all(mc, t) = mean(abs(ls_channel(:, t) - true_channel(:, t)).^2);
    end
    
    % 記錄平均MSE
    gmee_avg_mse(mc) = mean(gmee_mse_all(mc, :));
    ls_avg_mse(mc) = mean(ls_mse_all(mc, :));
    
    % 進度顯示
    if mod(mc, 10) == 0
        elapsed = toc;
        eta = elapsed / mc * (N_monte - mc);
        fprintf('完成: %d/%d (%.1f%%), 耗時: %.1fs, 預估剩餘: %.1fs\n', ...
            mc, N_monte, mc/N_monte*100, elapsed, eta);
    end
end

total_time = toc;
fprintf('\n模擬完成! 總耗時: %.2f 秒\n', total_time);

%% 4. 統計分析
fprintf('\n=== 統計分析結果 ===\n');

% 平均MSE
gmee_mean = mean(gmee_avg_mse);
ls_mean = mean(ls_avg_mse);
gmee_std = std(gmee_avg_mse);
ls_std = std(ls_avg_mse);

fprintf('GMEE-EKF: 平均MSE = %.6f ± %.6f\n', gmee_mean, gmee_std);
fprintf('LS:       平均MSE = %.6f ± %.6f\n', ls_mean, ls_std);

improvement_mean = (ls_mean - gmee_mean) / ls_mean * 100;
fprintf('平均改善: %.2f%%\n', improvement_mean);

% 時間平均的MSE曲線
gmee_mse_mean = mean(gmee_mse_all, 1);
ls_mse_mean = mean(ls_mse_all, 1);
gmee_mse_std = std(gmee_mse_all, 0, 1);
ls_mse_std = std(ls_mse_all, 0, 1);

% 改善百分比
improvement_per_symbol = (ls_mse_mean - gmee_mse_mean) ./ ls_mse_mean * 100;

% 勝率統計
win_count = sum(gmee_avg_mse < ls_avg_mse);
win_rate = win_count / N_monte * 100;
fprintf('GMEE勝率: %.1f%% (%d/%d)\n', win_rate, win_count, N_monte);

%% 5. 視覺化結果
figure('Position', [50, 50, 1600, 900]);

% 1. MSE時間演進 (帶標準差)
subplot(2,3,1);
fill([1:Nsym, fliplr(1:Nsym)], ...
     [gmee_mse_mean - gmee_mse_std, fliplr(gmee_mse_mean + gmee_mse_std)], ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
hold on;
fill([1:Nsym, fliplr(1:Nsym)], ...
     [ls_mse_mean - ls_mse_std, fliplr(ls_mse_mean + ls_mse_std)], ...
     'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
plot(1:Nsym, gmee_mse_mean, 'r-', 'LineWidth', 2);
plot(1:Nsym, ls_mse_mean, 'g-', 'LineWidth', 2);
grid on;
xlabel('OFDM符號索引');
ylabel('MSE');
title(sprintf('MSE時間演進 (N=%d次平均)', N_monte));
legend('GMEE ±σ', 'LS ±σ', 'GMEE均值', 'LS均值', 'Location', 'best');

% 2. MSE時間演進 (對數尺度)
subplot(2,3,2);
semilogy(1:Nsym, gmee_mse_mean, 'r-', 'LineWidth', 2);
hold on;
semilogy(1:Nsym, ls_mse_mean, 'g-', 'LineWidth', 2);
grid on;
xlabel('OFDM符號索引');
ylabel('MSE (對數)');
title('MSE演進 (對數尺度)');
legend('GMEE-EKF', 'LS', 'Location', 'best');

% 3. 改善百分比隨時間變化
subplot(2,3,3);
plot(1:Nsym, improvement_per_symbol, 'g-', 'LineWidth', 2);
hold on;
yline(mean(improvement_per_symbol), 'r--', 'LineWidth', 1.5);
grid on;
xlabel('OFDM符號索引');
ylabel('改善百分比 (%)');
title(sprintf('每符號改善 (平均: %.2f%%)', mean(improvement_per_symbol)));
ylim([min(improvement_per_symbol)-5, max(improvement_per_symbol)+5]);

% 4. 平均MSE分佈 (直方圖)
subplot(2,3,4);
histogram(gmee_avg_mse, 20, 'FaceColor', 'r', 'FaceAlpha', 0.6, 'EdgeColor', 'k');
hold on;
histogram(ls_avg_mse, 20, 'FaceColor', 'b', 'FaceAlpha', 0.6, 'EdgeColor', 'k');
xline(gmee_mean, 'r--', 'LineWidth', 2);
xline(ls_mean, 'g--', 'LineWidth', 2);
grid on;
xlabel('平均MSE');
ylabel('次數');
title('MSE分佈直方圖');
legend('GMEE-EKF', 'LS', 'GMEE均值', 'LS均值', 'Location', 'best');

% 5. 累積分佈函數 (CDF)
subplot(2,3,5);
[f_gmee, x_gmee] = ecdf(gmee_avg_mse);
[f_ls, x_ls] = ecdf(ls_avg_mse);
plot(x_gmee, f_gmee, 'r-', 'LineWidth', 2);
hold on;
plot(x_ls, f_ls, 'g-', 'LineWidth', 2);
grid on;
xlabel('平均MSE');
ylabel('累積機率');
title('MSE累積分佈函數');
legend('GMEE-EKF', 'LS', 'Location', 'best');


%% 6. 額外統計圖表
figure('Position', [100, 100, 1400, 500]);

% 1. 每次模擬的改善百分比

improvement_all = (ls_avg_mse - gmee_avg_mse) ./ ls_avg_mse * 100;
plot(1:N_monte, improvement_all, 'g.-', 'MarkerSize', 8);
hold on;
yline(mean(improvement_all), 'r--', 'LineWidth', 2);
yline(0, 'k-', 'LineWidth', 1);
grid on;
xlabel('模擬次數');
ylabel('改善百分比 (%)');
title(sprintf('每次模擬的改善 (平均: %.2f%%)', mean(improvement_all)));





%% 7. 詳細統計報告
fprintf('\n=== 詳細統計報告 ===\n');
fprintf('--------------------------------------------------\n');
fprintf('演算法     | 均值MSE    | 標準差      | 中位數      \n');
fprintf('--------------------------------------------------\n');
fprintf('GMEE-EKF   | %.6f | %.6f | %.6f\n', gmee_mean, gmee_std, median(gmee_avg_mse));
fprintf('LS         | %.6f | %.6f | %.6f\n', ls_mean, ls_std, median(ls_avg_mse));
fprintf('--------------------------------------------------\n');
fprintf('改善統計:\n');
fprintf('  平均改善: %.2f%%\n', improvement_mean);
fprintf('  最大改善: %.2f%%\n', max(improvement_all));
fprintf('  最小改善: %.2f%%\n', min(improvement_all));
fprintf('  改善標準差: %.2f%%\n', std(improvement_all));
fprintf('  勝率: %.1f%% (%d/%d)\n', win_rate, win_count, N_monte);
fprintf('--------------------------------------------------\n');

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
    
    % GMEE參數
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