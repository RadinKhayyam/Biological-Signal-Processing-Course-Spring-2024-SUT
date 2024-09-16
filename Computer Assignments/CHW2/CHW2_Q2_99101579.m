%% BSP - CHW2 - 99101579
%% Q2 - Part a
clc; clear; close all;

fs = 360;
length_sig = 5; % 5 seconds signal
gain = 200;
ecg_signal = load("ECG.mat").val(1:length_sig*fs)/gain;

t = 0:1/fs:length_sig-1/fs;    
frequency = 50;
amplitude = 0.2;

phase1 = rand() * 2 * pi;
phase2 = rand() * 2 * pi;

noise1 = amplitude * sin(2 * pi * frequency * t + phase1);
noise2 = amplitude * sin(2 * pi * frequency * t + phase2); % Reference signal

noisy_ecg_signal = ecg_signal + noise1; % Primary signal

figure;
subplot(2, 1, 1);
plot(t, ecg_signal,'LineWidth',1);
title('Original ECG Signal','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex','FontSize',10);
ylabel('Amplitude','Interpreter','latex','FontSize',10);

subplot(2, 1, 2);
plot(t, noisy_ecg_signal,'LineWidth',1);
title('Noisy ECG Signal','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex','FontSize',10);
ylabel('Amplitude','Interpreter','latex','FontSize',10);

M = 15; %Filter Order
mu = 0.01;

[output_signal, error_signal, w] = myAdaptiveFilter(M, mu, noisy_ecg_signal, noise2);

%% Q2 - Part b
clc;
figure;
plot(t, noisy_ecg_signal);
hold on;
plot(t(M+1:end), error_signal(M+1:end),'LineWidth',1);
title('Noisy ECG Signal and Adaptive Filter Output','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex','FontSize',10);
ylabel('Amplitude','Interpreter','latex','FontSize',10);
legend('Noisy ECG Signal', 'Adaptive Filter Output');
hold off;

% Spectral analysis using pwelch
[pxx_noisy, f] = pwelch(noisy_ecg_signal, [], [], [], fs);
[pxx_denoised, f] = pwelch(error_signal, [], [], [], fs);
[pxx_ecg, f] = pwelch(ecg_signal, [], [], [], fs);

figure;
plot(f, pxx_noisy,'LineWidth',1);
hold on;
plot(f, pxx_denoised,'LineWidth',1);
title('Power Spectral Density of Noisy Signal and Denoised Signal','Interpreter','latex','FontSize',14);
xlabel('Frequency (Hz)','Interpreter','latex','FontSize',10);
ylabel('Power/Frequency (1/Hz)','Interpreter','latex','FontSize',10);
legend('Noisy Signal', 'Denoised Signal');
xlim([3 fs/2]);
hold off;


%% Q2 - Part c
clc;
noise_amplitudes = 0.1:0.01:1; % Range of noise amplitudes
snr_improvement = zeros(size(noise_amplitudes));
snr_input = zeros(size(noise_amplitudes));
snr_output = zeros(size(noise_amplitudes));

M = 20;
mu = 0.01;
for i = 1:length(noise_amplitudes)

    amplitude = noise_amplitudes(i);
    phase1 = rand() * 2 * pi;
    phase2 = rand() * 2 * pi;
    noise1 = amplitude * sin(2 * pi * frequency * t + phase1);
    noise2 = amplitude * sin(2 * pi * frequency * t + phase2);
    
    noisy_ecg_signal = ecg_signal + noise1;
    
    [output_signal, error_signal, w] = myAdaptiveFilter(M, mu, noisy_ecg_signal, noise2);
    
    snr_input(i) = calculate_snr(ecg_signal(M+1:end), noise1(M+1:end));
    snr_output(i) = calculate_snr(ecg_signal(M+1:end), error_signal(M+1:end) - ecg_signal(M+1:end));
    snr_improvement(i) = snr_output(i) - snr_input(i);

end

figure;
plot(snr_input, snr_improvement,'LineWidth',2);
title('SNR Improvement vs SNR Input','Interpreter','latex','FontSize',14);
xlabel('SNR Input (dB)','Interpreter','latex','FontSize',10);
ylabel('SNR Improvement (dB)','Interpreter','latex','FontSize',10);
grid minor;

%% Q2 - Part d
clc;

fs = 360;
length_sig = 5; % 5 seconds signal
gain = 200;
ecg_signal = load("ECG.mat").val(1:length_sig*fs)/gain;

t = 0:1/fs:length_sig-1/fs;    
frequency = 50;
amplitude = 0.2;

phase1 = rand() * 2 * pi;
phase2 = rand() * 2 * pi;

noise1 = amplitude * sin(2 * pi * frequency * t + phase1);
noise2 = amplitude * sin(2 * pi * frequency * t + phase2); % Reference signal

noisy_ecg_signal = ecg_signal + noise1; % Primary signal

M = 10; %Filter Order
mu = 0.01;

[output_signal, error_signal, w] = myAdaptiveFilter(M, mu, noisy_ecg_signal, noise2);

impulse_response = filter(1, [1; -w], [1; zeros(length(t)-1, 1)]);

[h, f] = freqz([1; -w], 1, 512, fs);

figure;
subplot(3, 1, 1);
stem(impulse_response);
title('Impulse Response of Stop Band Filter');
xlabel('Sample');
ylabel('Amplitude');

subplot(3, 1, 2);
plot(f, 20*log10(abs(h)));
title('Frequency Response of Stop Band Filter');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
grid on;

subplot(3, 1, 3);
zplane([1; -w], 1);
title('Pole-Zero Plot of Stop Band Filter');


%% Q2 - Part e
clc;

amplitude = 0.2;
center_frequency = 50;  
frequency_fluctuation = 5;  
% noise_frequency = center_frequency + frequency_fluctuation * rand(size(t));
noise_frequency = center_frequency + frequency_fluctuation * t;

phase1 = rand() * 2 * pi;
phase2 = rand() * 2 * pi;

noise1 = amplitude * sin(2 * pi * noise_frequency .* t + phase1);
noise2 = amplitude * sin(2 * pi * noise_frequency .* t + phase2);

noisy_ecg_signal = ecg_signal + noise1;

M = 10;
mu = 0.1;

[output_signal, error_signal, w] = myAdaptiveFilter(M, mu, noisy_ecg_signal, noise2);


figure;
plot(t, noisy_ecg_signal);
hold on;
plot(t(M+1:end), error_signal(M+1:end),'LineWidth',1);
title('Noisy ECG Signal and Adaptive Filter Output','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex','FontSize',10);
ylabel('Amplitude','Interpreter','latex','FontSize',10);
legend('Noisy ECG Signal', 'Adaptive Filter Output');
hold off;

[pxx_noisy_harmonic, f] = pwelch(noisy_ecg_signal, [], [], [], fs);
[pxx_denoised_harmonic, f] = pwelch(error_signal, [], [], [], fs);
[pxx_ecg, f] = pwelch(ecg_signal, [], [], [], fs);

figure;
plot(f, pxx_noisy_harmonic,'LineWidth',1);
hold on;
plot(f, pxx_denoised_harmonic,'LineWidth',1);
title('Power Spectral Density of Noisy Signal and Denoised Signal','Interpreter','latex','FontSize',14);
xlabel('Frequency (Hz)','Interpreter','latex','FontSize',10);
ylabel('Power/Frequency (1/Hz)','Interpreter','latex','FontSize',10);
legend('Noisy Signal', 'Denoised Signal');
xlim([3 fs/2]);
hold off;


%% Q2 - Part f
clc;
amplitude = 0.2;
noise_frequency = 50;

phase1 = rand() * 2 * pi;
phase2 = rand() * 2 * pi;

noise1_harmonic = amplitude * sin(2 * pi * noise_frequency * t + phase1) + 0.25 * amplitude * sin(2 * pi * 2 * noise_frequency * t + phase1);
noise2_harmonic = amplitude * sin(2 * pi * noise_frequency * t + phase2) + 0.25 * amplitude * sin(2 * pi * 2 * noise_frequency * t + phase2);

noise1 = amplitude * sin(2 * pi * noise_frequency * t + phase1);
noise2 = amplitude * sin(2 * pi * noise_frequency * t + phase2);

noisy_ecg_signal_harmonic = ecg_signal + noise1_harmonic;
noisy_ecg_signal = ecg_signal + noise1;

M = 10;
mu = 0.01;

[output_signal_harmonic, error_signal_harmonic, w_harmonic] = myAdaptiveFilter(M, mu, noisy_ecg_signal_harmonic, noise2_harmonic);
[output_signal, error_signal, w] = myAdaptiveFilter(M, mu, noisy_ecg_signal, noise2);

snr_input_harmonic = calculate_snr(ecg_signal(M+1:end), noise1_harmonic(M+1:end));
snr_output_harmonic = calculate_snr(ecg_signal(M+1:end), error_signal_harmonic(M+1:end) - ecg_signal(M+1:end));
snr_improvement_harmonic = snr_output_harmonic - snr_input_harmonic;

snr_input = calculate_snr(ecg_signal(M+1:end), noise1(M+1:end));
snr_output_harmonic = calculate_snr(ecg_signal(M+1:end), error_signal(M+1:end) - ecg_signal(M+1:end));
snr_improvement = snr_output_harmonic - snr_input_harmonic;

disp(['SNR improvement for signal with first harmonic = ',num2str(snr_improvement_harmonic)]);
disp(['SNR improvement for harmonic free signal = ',num2str(snr_improvement)]);

figure;
plot(t, noisy_ecg_signal);
hold on;
plot(t(M+1:end), error_signal(M+1:end),'LineWidth',1);
title('Noisy ECG Signal and Adaptive Filter Output','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex','FontSize',10);
ylabel('Amplitude','Interpreter','latex','FontSize',10);
legend('Noisy ECG Signal', 'Adaptive Filter Output');
hold off;

%% Q2 - Part g
clc;


amplitude = 0.2;
noise_frequency = 50;
noise1 = amplitude * sin(2 * pi * noise_frequency .* t);

noisy_ecg_signal = ecg_signal + noise1;

M = 10;
mu = 0.01;

delays = [1, 5, 10, 20, 50];
snr_improvements = zeros(length(delays), 1);

for i = 1:length(delays)
    delay = delays(i);
    delayed_signal = [zeros(1, delay), noisy_ecg_signal(1:end-delay)];
    
    [~, error_signal, ~] = myAdaptiveFilter(M, mu, noisy_ecg_signal, delayed_signal);
    
    snr_input = calculate_snr(ecg_signal(M+1:end), noise1(M+1:end));
    snr_output_harmonic = calculate_snr(ecg_signal(M+1:end), error_signal(M+1:end) - ecg_signal(M+1:end));
    snr_improvement = snr_output_harmonic - snr_input_harmonic;
    
    fprintf('SNR Improvement for delay = %d: %.2f dB\n', delay, snr_improvement);
end

figure;
plot(delays, snr_improvements, '-o');
title('SNR Improvement vs. Delay');
xlabel('Delay (samples)');
ylabel('SNR Improvement (dB)');
grid on;






%% Functions
function [output_signal, error_signal, w] = myAdaptiveFilter(M, mu, primary_signal, reference_signal)
    % LMS algorithm
    w = rand(M, 1);
    y = zeros(size(primary_signal));
    e = zeros(size(primary_signal));
    
    for n = M+1:length(primary_signal)
        x = reference_signal(n:-1:n-M+1)';  
        y(n) = w' * x;               % Filter output
        e(n) = primary_signal(n) - y(n);  % Error signal
        w = w + 2 * mu * e(n) * x;   % Update filter weights
    end  
    output_signal = y;
    error_signal = e;
end

function snr_value = calculate_snr(signal, noise)
    signal_power = mean(signal.^2);
    noise_power = mean(noise.^2);
    snr_value = 10 * log10(signal_power / noise_power);
end



