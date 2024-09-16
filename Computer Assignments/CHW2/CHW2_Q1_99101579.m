%% BSP - CHW2 - 99101579
%% Q1
clc; clear; close all;


alpha_1 = 2*pi*rand; 
alpha_2 = 2*pi*rand; 
alpha_3 = 2*pi*rand; 
n = 0:999;
x = 10*cos(0.1*pi*n + alpha_1) + 20*cos(0.4*pi*n + alpha_2) + 10*cos(0.8*pi*n + alpha_3) + randn(1, 1000);
%% Q1 - part a
clc;

omega_real = -pi:2*pi/2000:pi-(2*pi/2000);
S_x_real = zeros(1,2000);
S_x_real(1100) = 50*pi;
S_x_real(900) = 50*pi;
S_x_real(1400) = 200*pi;
S_x_real(600) = 200*pi;
S_x_real(1800) = 50*pi;
S_x_real(200) = 50*pi;
S_x_real(1000) = 1;

plot(omega_real, S_x_real,'LineWidth',1);
title('$$S_{x}(\omega)$$ - Real','Interpreter','latex','FontSize',20);
ylabel('Amplitude','Interpreter','latex','FontSize',14);
xlabel('$$\omega$$','Interpreter','latex','FontSize',14);
xlim([-pi pi]);
xticks([-pi -(8/10)*pi -(4/10)*pi -(1/10)*pi 0 (1/10)*pi (4/10)*pi  (8/10)*pi pi]);
xticklabels({'-\pi','-8/10 \pi','-4/10\pi','-1/10\pi','0', '1/10\pi','4/10\pi','8/10\pi','\pi','interpreter','latex','FontSize',10});
%% Q1 - part b
clc;
M = 1000;
Rx_est = zeros(1,2*M-1);
for m=0:M-1
    tmp = 0;
    for n=0:M-m-1
        tmp = tmp + x(n+1)*x(n+m+1);
    end
    Rx_est(m+1)=1/M * tmp;
    if(m~=0)
            Rx_est(2*M-m) = Rx_est(m+1);
    end  
end
s_x_BT = fftshift(fft(Rx_est));
L = length(s_x_BT);
omega_BT = -pi:2*pi/L:pi-(2*pi/L);

subplot(2,1,1);
plot(omega_real, S_x_real,'LineWidth',1);
title('$$S_{x}(\omega)$$ - Real','Interpreter','latex','FontSize',20);
ylabel('Amplitude','Interpreter','latex','FontSize',14);
xlabel('$$\omega$$','Interpreter','latex','FontSize',14);
xlim([-pi pi]);
xticks([-pi -(8/10)*pi -(4/10)*pi -(1/10)*pi 0 (1/10)*pi (4/10)*pi  (8/10)*pi pi]);
xticklabels({'-\pi','-8/10 \pi','-4/10\pi','-1/10\pi','0', '1/10\pi','4/10\pi','8/10\pi','\pi','interpreter','latex','FontSize',10});

subplot(2,1,2);
plot(omega_BT, s_x_BT,'Linewidth',1);
title('$$S_{x}(\omega)$$ - BT Estimation','Interpreter','latex','FontSize',20);
ylabel('Amplitude','Interpreter','latex','FontSize',14);
xlabel('$$\omega$$','Interpreter','latex','FontSize',14);
xlim([-pi pi]);
xticks([-pi -(8/10)*pi -(4/10)*pi -(1/10)*pi 0 (1/10)*pi (4/10)*pi  (8/10)*pi pi]);
xticklabels({'-\pi','-8/10 \pi','-4/10\pi','-1/10\pi','0', '1/10\pi','4/10\pi','8/10\pi','\pi','interpreter','latex','FontSize',10});        
%% Q1 - part c
clc;
W_1 = rectwin(M)';
W_2 = hamming(M)';
W_3 = gausswin(M)';

Xw_1 = W_1.*x;
Xw_2 = W_2.*x;
Xw_3 = W_3.*x;

Xw_1_fourier = fftshift(fft(Xw_1));
Xw_2_fourier = fftshift(fft(Xw_2));
Xw_3_fourier = fftshift(fft(Xw_3));

Sx_1_peridogram = 1/M * abs(Xw_1_fourier).^2;
Sx_2_peridogram = 1/M * abs(Xw_2_fourier).^2;
Sx_3_peridogram = 1/M * abs(Xw_3_fourier).^2;

omega_periodogram = -pi:2*pi/M:pi-(2*pi/M);

subplot(4,1,1);
plot(omega_real, S_x_real,'LineWidth',1);
title('$$S_{x}(\omega)$$ - Real','Interpreter','latex','FontSize',16);
ylabel('Amplitude','Interpreter','latex','FontSize',14);
xlabel('$$\omega$$','Interpreter','latex','FontSize',14);
xlim([-pi pi]);
xticks([-pi -(8/10)*pi -(4/10)*pi -(1/10)*pi 0 (1/10)*pi (4/10)*pi  (8/10)*pi pi]);
xticklabels({'-\pi','-8/10 \pi','-4/10\pi','-1/10\pi','0', '1/10\pi','4/10\pi','8/10\pi','\pi','interpreter','latex','FontSize',10});

subplot(4,1,2);
plot(omega_periodogram, Sx_1_peridogram,'LineWidth',1);
title('$$S_{x}(\omega)$$ - Periodogram - Rect Window','Interpreter','latex','FontSize',16);
ylabel('Amplitude','Interpreter','latex','FontSize',14);
xlabel('$$\omega$$','Interpreter','latex','FontSize',14);
xlim([-pi pi]);
xticks([-pi -(8/10)*pi -(4/10)*pi -(1/10)*pi 0 (1/10)*pi (4/10)*pi  (8/10)*pi pi]);
xticklabels({'-\pi','-8/10 \pi','-4/10\pi','-1/10\pi','0', '1/10\pi','4/10\pi','8/10\pi','\pi','interpreter','latex','FontSize',10}); 

subplot(4,1,3);
plot(omega_periodogram, Sx_2_peridogram,'LineWidth',1);
title('$$S_{x}(\omega)$$ - Periodogram - Hamming Window','Interpreter','latex','FontSize',16);
ylabel('Amplitude','Interpreter','latex','FontSize',14);
xlabel('$$\omega$$','Interpreter','latex','FontSize',14);
xlim([-pi pi]);
xticks([-pi -(8/10)*pi -(4/10)*pi -(1/10)*pi 0 (1/10)*pi (4/10)*pi  (8/10)*pi pi]);
xticklabels({'-\pi','-8/10 \pi','-4/10\pi','-1/10\pi','0', '1/10\pi','4/10\pi','8/10\pi','\pi','interpreter','latex','FontSize',10}); 

subplot(4,1,4);
plot(omega_periodogram, Sx_3_peridogram,'LineWidth',1);
title('$$S_{x}(\omega)$$ - Periodogram - Gaussian Window','Interpreter','latex','FontSize',16);
ylabel('Amplitude','Interpreter','latex','FontSize',14);
xlabel('$$\omega$$','Interpreter','latex','FontSize',14);
xlim([-pi pi]);
xticks([-pi -(8/10)*pi -(4/10)*pi -(1/10)*pi 0 (1/10)*pi (4/10)*pi  (8/10)*pi pi]);
xticklabels({'-\pi','-8/10 \pi','-4/10\pi','-1/10\pi','0', '1/10\pi','4/10\pi','8/10\pi','\pi','interpreter','latex','FontSize',10}); 

%% Q1 - Part d
clc;
window_lengths = [50 100 500];
nfft = 256; 
overlaps = [10, 20, 30];
psds_fix_wl = [];
psds_fix_ol = [];

for i = 1:3
    psds_fix_wl = [psds_fix_wl, fftshift(pwelch(x,window_lengths(2),overlaps(i),nfft,'twosided'))];
end
for i = 1:3
    psds_fix_ol = [psds_fix_ol, fftshift(pwelch(x,window_lengths(i),overlaps(3),nfft,'twosided'))];
end
L = nfft;
omega_welch = -pi:2*pi/L:pi-2*pi/L;

subplot(4,1,1);
plot(omega_real, S_x_real,'LineWidth',1);
title('$$S_{x}(\omega)$$ - Real','Interpreter','latex','FontSize',16);
ylabel('Amplitude','Interpreter','latex','FontSize',14);
xlabel('$$\omega$$','Interpreter','latex','FontSize',14);
xlim([-pi pi]);
xticks([-pi -(8/10)*pi -(4/10)*pi -(1/10)*pi 0 (1/10)*pi (4/10)*pi  (8/10)*pi pi]);
xticklabels({'-\pi','-8/10 \pi','-4/10\pi','-1/10\pi','0', '1/10\pi','4/10\pi','8/10\pi','\pi','interpreter','latex','FontSize',10});

for i=1:3
    subplot(4,1,i+1)
    plot(omega_welch,psds_fix_wl(:,i),'LineWidth',1);
    title(['$$S_{x}(\omega)$$ - Welch - Window Length=100 - Overlap=',num2str(overlaps(i))],'Interpreter','latex','FontSize',16);
    ylabel('Amplitude','Interpreter','latex','FontSize',14);
    xlabel('$$\omega$$','Interpreter','latex','FontSize',14);
    xlim([-pi pi]);
    xticks([-pi -(8/10)*pi -(4/10)*pi -(1/10)*pi 0 (1/10)*pi (4/10)*pi  (8/10)*pi pi]);
    xticklabels({'-\pi','-8/10 \pi','-4/10\pi','-1/10\pi','0', '1/10\pi','4/10\pi','8/10\pi','\pi','interpreter','latex','FontSize',10}); 
    hold on
end

figure;
subplot(4,1,1);
plot(omega_real, S_x_real,'LineWidth',1);
title('$$S_{x}(\omega)$$ - Real','Interpreter','latex','FontSize',16);
ylabel('Amplitude','Interpreter','latex','FontSize',14);
xlabel('$$\omega$$','Interpreter','latex','FontSize',14);
xlim([-pi pi]);
xticks([-pi -(8/10)*pi -(4/10)*pi -(1/10)*pi 0 (1/10)*pi (4/10)*pi  (8/10)*pi pi]);
xticklabels({'-\pi','-8/10 \pi','-4/10\pi','-1/10\pi','0', '1/10\pi','4/10\pi','8/10\pi','\pi','interpreter','latex','FontSize',10});

for i=1:3
    subplot(4,1,i+1)
    plot(omega_welch,psds_fix_ol(:,i),'LineWidth',1);
    title(['$$S_{x}(\omega)$$ - Welch - Window Length=',num2str(window_lengths(i)),' - overlap=30'],'Interpreter','latex','FontSize',16);
    ylabel('Amplitude','Interpreter','latex','FontSize',14);
    xlabel('$$\omega$$','Interpreter','latex','FontSize',14);
    xlim([-pi pi]);
    xticks([-pi -(8/10)*pi -(4/10)*pi -(1/10)*pi 0 (1/10)*pi (4/10)*pi  (8/10)*pi pi]);
    xticklabels({'-\pi','-8/10 \pi','-4/10\pi','-1/10\pi','0', '1/10\pi','4/10\pi','8/10\pi','\pi','interpreter','latex','FontSize',10}); 
    hold on
end

%% Q1 - Part e
clc;
M = 1000;
Rx = zeros(1,M);
for m=0:M-1
    tmp = 0;
    for n=0:M-m-1
        tmp = tmp + x(n+1)*x(n+m+1);
    end
    Rx(m+1)=1/M * tmp; 
end

max_order = 50;  
AIC = zeros(1, max_order);  
E = zeros(1, max_order);  

% Compute AR coefficients and AIC for each model order
for p = 1:max_order
    [a, e] = levinson(Rx(1:p+1), p); 
    AIC(p) = M * log(e) + 2 * (p + 1);  
    E(p)= e;
end

% Plot AIC values
figure;
subplot(1,2,1);
plot(1:max_order, E, '-o','LineWidth',1);
title('$$E_k$$ for Different AR Model Orders - Levinson-Durbin','Interpreter','latex','FontSize',16);
xlabel('Model Order(k)','Interpreter','latex','FontSize',14);
ylabel('$$E_k$$ Value','Interpreter','latex','FontSize',14);
grid minor;
subplot(1,2,2);
plot(1:max_order, AIC, '-o','LineWidth',1);
title('AIC for Different AR Model Orders - Levinson-Durbin','Interpreter','latex','FontSize',16);
xlabel('Model Order','Interpreter','latex','FontSize',14);
ylabel('AIC Value','Interpreter','latex','FontSize',14);
grid minor;

[~, optimal_order] = min(AIC);
fprintf('Optimal AR Model Order: %d\n', optimal_order);


methods = {'Levinson-Durbin', 'Yule-Walker', 'Burg', 'Covariance', 'Modified Covariance'};
coeffs = cell(length(methods), 1);
psd = cell(length(methods), 1);


omega = -pi:2*pi/512:pi-2*pi/512;
figure;
for i = 1:5
    switch methods{i}
        case 'Levinson-Durbin'
            [a_est, ~] = levinson(Rx, optimal_order);
        case 'Yule-Walker'
            a_est = aryule(x, optimal_order);
        case 'Burg'
            a_est = arburg(x, optimal_order);
        case 'Covariance'
            a_est = arcov(x, optimal_order);
        case 'Modified Covariance'
            a_est = armcov(x, optimal_order);
    end
   
    coeffs{i} = a_est;
    
    [h, w] = freqz(1, a_est, 512, 'whole');
    psd{i} = fftshift(abs(h).^2);
    
    subplot(5, 2, 2*i-1);
    stem(a_est, 'filled');
    title([methods{i} ' Coefficients'],'Interpreter','latex');
    xlabel('Coefficient Index','Interpreter','latex');
    ylabel('Value','Interpreter','latex');
    grid on;
    
    subplot(5, 2, 2*i);
    plot(omega, psd{i});
    title([methods{i} ' PSD'],'Interpreter','latex');
    xlabel('$$\omega$$','Interpreter','latex');
    ylabel('Amplitude','Interpreter','latex');
    xticks([-pi -(8/10)*pi -(4/10)*pi -(1/10)*pi (1/10)*pi (4/10)*pi  (8/10)*pi pi]);
    xticklabels({'-\pi','-8/10 \pi','-4/10\pi','-1/10\pi', '1/10\pi','4/10\pi','8/10\pi','\pi','interpreter','latex','FontSize',10});
    grid on;
end

%% Q1 - Part f
clc;


max_order = 10; 

AIC = zeros(maxOrder, 1);
coeffsArray = cell(max_order, 1);

for q = 1:max_order
    M = zeros(q+1, q+1);
    b = zeros(q+1, 1);

    for m = 0:q
        b(m+1, 1) = Rx(m+1); 
        for k = 0:q
            if m-k >= 0
                M(m+1, k+1) = Rx(abs(m-k)+1);
            else
                M(m+1, k+1) = Rx(abs(k-m)+1);
            end
        end
    end
    

    coeffs = M \ b; 
    

    coeffsArray{q} = coeffs(2:end) / coeffs(1);
    
    e = filter([1; -coeffsArray{q}], 1, x);

    sigma2 = var(e);
    N = length(x);
    AIC(q) = N * log(sigma2) + 2 * (q + 1); 
end

[~, optimal_order] = min(AIC);

fprintf('Optimal MA Model Order: %d\n', optimal_order);
fprintf('Minimum AIC: %.2f\n', minAIC);
fprintf('Optimal MA Coefficients:');
disp(coeffsArray{optimal_order});

plot(1:max_order, AIC, '-o','LineWidth',1);
title('AIC for Different MA Model Orders - Levinson-Durbin','Interpreter','latex','FontSize',16);
xlabel('Model Order','Interpreter','latex','FontSize',14);
ylabel('AIC Value','Interpreter','latex','FontSize',14);
grid minor;

%%

