%% BSP - CHW1
%% Radin Khayyam - 99101579
%% Question 1
clc; clear; close all;
data_1 = load('EEG_990mS16.mat');
data_2 = load('EEG_2700mS16.mat');
fs = 256; % sampling frequency
%% Q1 - Part a
clc;
EEG_1_C3 = data_1.y(:,:,1);
EEG_1_C4 = data_1.y(:,:,2);
EEG_2_C3 = data_2.y(:,:,1);
EEG_2_C4 = data_2.y(:,:,2);


L = length(EEG_1_C3)/fs; % length of signal (s)
t = 0:1/fs:L-1/fs; % time axes
trial_slct = 5; % selected trial

figure;
subplot(4,1,1);
plot(t,EEG_1_C3(trial_slct, :));
title(['C3 signal at 990m height - trial number ',num2str(trial_slct)], 'Interpreter','latex','FontSize',14);
xlabel("time (s)",'Interpreter','latex');
ylabel("$\mu$ Volt","Interpreter",'latex');

subplot(4,1,2);
plot(t,EEG_2_C3(trial_slct, :));
title(['C3 signal at 2700m height - trial number ',num2str(trial_slct)], 'Interpreter','latex','FontSize',14);
xlabel("time (s)",'Interpreter','latex');
ylabel("$\mu$ Volt","Interpreter",'latex');

subplot(4,1,3);
plot(t,EEG_1_C4(trial_slct, :));
title(['C4 signal at 990m height - trial number ',num2str(trial_slct)], 'Interpreter','latex','FontSize',14);
xlabel("time (s)",'Interpreter','latex');
ylabel("$\mu$ Volt","Interpreter",'latex');

subplot(4,1,4);
plot(t,EEG_2_C4(trial_slct, :));
title(['C4 signal at 2700m height - trial number ',num2str(trial_slct)], 'Interpreter','latex','FontSize',14);
xlabel("time (s)",'Interpreter','latex');
ylabel("$\mu$ Volt","Interpreter",'latex');

%% Q1 - Part b
clc; close all;

NFFT = 2^nextpow2(length(EEG_2_C4(trial_slct,:)));
fft_1_C3 = fft(EEG_1_C3(trial_slct,:),NFFT);
fft_2_C3 = fft(EEG_2_C3(trial_slct,:),NFFT);
fft_1_C4 = fft(EEG_1_C4(trial_slct,:),NFFT);
fft_2_C4 = fft(EEG_2_C4(trial_slct,:),NFFT);

f = fs/NFFT*(-NFFT/2:NFFT/2-1); % frequency axes

figure;
subplot(4,2,1);
plot(f,abs(fftshift(fft_1_C3)));
title(['Amplitude of DTFT of C3 signal at 990m - trial number',num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel("frequency (Hz)",'Interpreter','latex');
ylabel("|DTFT(X)|",'Interpreter','latex');
xlim([-fs/2,fs/2]);
subplot(4,2,2);
plot(f,180/pi*angle(fftshift(fft_1_C3)));
title(['Phase of DTFT of C3 signal at 990m - trial number',num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel("frequency (Hz)",'Interpreter','latex');
ylabel("$\angle$DTFT(X)",'Interpreter','latex');
xlim([-fs/2,fs/2]);
ylim([-180,180]);

subplot(4,2,3);
plot(f,abs(fftshift(fft_2_C3)));
title(['Amplitude of DTFT of C3 signal at 2700m - trial number',num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel("frequency (Hz)",'Interpreter','latex');
ylabel("|DTFT(X)|",'Interpreter','latex');
xlim([-fs/2,fs/2]);
subplot(4,2,4);
plot(f,180/pi*angle(fftshift(fft_2_C3)));
title(['Phase of DTFT of C3 signal at 2700m - trial number',num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel("frequency (Hz)",'Interpreter','latex');
ylabel("$\angle$DTFT(X)",'Interpreter','latex');
xlim([-fs/2,fs/2]);
ylim([-180,180]);

subplot(4,2,5);
plot(f,abs(fftshift(fft_1_C4)));
title(['Amplitude of DTFT of C4 signal at 990m - trial number',num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel("frequency (Hz)",'Interpreter','latex');
ylabel("|DTFT(X)|",'Interpreter','latex');
xlim([-fs/2,fs/2]);
subplot(4,2,6);
plot(f,180/pi*angle(fftshift(fft_1_C4)));
title(['Phase of DTFT of C4 signal at 990m - trial number',num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel("frequency (Hz)",'Interpreter','latex');
ylabel("$\angle$DTFT(X)",'Interpreter','latex');
xlim([-fs/2,fs/2]);
ylim([-180,180]);

subplot(4,2,7);
plot(f,abs(fftshift(fft_2_C4)));
title(['Amplitude of DTFT of C4 signal at 2700m - trial number',num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel("frequency (Hz)",'Interpreter','latex');
ylabel("|DTFT(X)|",'Interpreter','latex');
xlim([-fs/2,fs/2]);
subplot(4,2,8);
plot(f,180/pi*angle(fftshift(fft_2_C4)));
title(['Phase of DTFT of C4 signal at 2700m - trial number',num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel("frequency (Hz)",'Interpreter','latex');
ylabel("$\angle$DTFT(X)",'Interpreter','latex');
xlim([-fs/2,fs/2]);
ylim([-180,180]);

%% Q1 - Part c
clc; close all;

NFFT = 2^nextpow2(length(EEG_2_C4(trial_slct,:)));
fft_1_C3 = fft(EEG_1_C3(trial_slct,:),NFFT);
fft_2_C3 = fft(EEG_2_C3(trial_slct,:),NFFT);
fft_1_C4 = fft(EEG_1_C4(trial_slct,:),NFFT);
fft_2_C4 = fft(EEG_2_C4(trial_slct,:),NFFT);

k = 0:1:NFFT-1;

figure;
subplot(4,2,1);
plot(k,abs(fft_1_C3));
title(['Amplitude of DTFT of C3 signal at 990m - trial number ',num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel("k",'Interpreter','latex');
ylabel("$\|$DTFT(X)$\|$",'Interpreter','latex');
xlim([0,NFFT-1]);
subplot(4,2,2);
plot(k,180/pi*angle(fft_1_C3));
title(['Phase of DTFT of C3 signal at 990m - trial number ',num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel("k",'Interpreter','latex');
ylabel("$\angle$DTFT(X)",'Interpreter','latex');
ylim([-180,180]);
xlim([0,NFFT-1]);

subplot(4,2,3);
plot(k,abs(fft_2_C3));
title(['Amplitude of DTFT of C3 signal at 2700m - trial number ',num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel("k",'Interpreter','latex');
ylabel("$\|$DTFT(X)$\|$",'Interpreter','latex');
xlim([0,NFFT-1]);
subplot(4,2,4);
plot(k,180/pi*angle(fft_2_C3));
title(['Phase of DTFT of C3 signal at 2700m - trial number ',num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel("k",'Interpreter','latex');
ylabel("$\angle$DTFT(X)",'Interpreter','latex');
ylim([-180,180]);
xlim([0,NFFT-1]);

subplot(4,2,5);
plot(k,abs(fft_1_C4));
title(['Amplitude of DTFT of C4 signal at 990m - trial number ',num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel("k",'Interpreter','latex');
ylabel("$\|$DTFT(X)$\|$",'Interpreter','latex');
xlim([0,NFFT-1]);
subplot(4,2,6);
plot(k,180/pi*angle(fft_1_C4));
title(['Phase of DTFT of C4 signal at 990m - trial number ',num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel("k",'Interpreter','latex');
ylabel("$\angle$DTFT(X)",'Interpreter','latex');
ylim([-180,180]);
xlim([0,NFFT-1]);

subplot(4,2,7);
plot(k,abs(fft_2_C4));
title(['Amplitude of DTFT of C4 signal at 2700m - trial number ',num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel("k",'Interpreter','latex');
ylabel("$\|$DTFT(X)$\|$",'Interpreter','latex');
xlim([0,NFFT-1]);
subplot(4,2,8);
plot(k,180/pi*angle(fft_2_C4));
title(['Phase of DTFT of C4 signal at 2700m - trial number ',num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel("k",'Interpreter','latex');
ylabel("$\angle$DTFT(X)",'Interpreter','latex');
ylim([-180,180]);
xlim([0,NFFT-1]);

%% Q1 - Part d
clc; close all;

[PSD_1_C3, ~] = pwelch(EEG_1_C3(trial_slct,:),[],[],NFFT,fs);
[PSD_2_C3, ~] = pwelch(EEG_2_C3(trial_slct,:),[],[],NFFT,fs);
[PSD_1_C4, ~] = pwelch(EEG_1_C4(trial_slct,:),[],[],NFFT,fs);
[PSD_2_C4, f] = pwelch(EEG_2_C4(trial_slct,:),[],[],NFFT,fs);

figure;
subplot(4,1,1);
plot(f,PSD_1_C3);
title(['PSD of C3 signal at 990m height - trial number ', num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel('Frequency (Hz)','Interpreter','latex');
ylabel('PSD (1/Hz)','Interpreter','latex');
xlim([0,fs/2]);
grid minor;

subplot(4,1,2);
plot(f,PSD_2_C3);
title(['PSD of C3 signal at 2700m height - trial number ', num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel('Frequency (Hz)','Interpreter','latex');
ylabel('PSD (1/Hz)','Interpreter','latex');
xlim([0,fs/2]);
grid minor;

subplot(4,1,3);
plot(f,PSD_1_C4);
title(['PSD of C4 signal at 990m height - trial number ', num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel('Frequency (Hz)','Interpreter','latex');
ylabel('PSD (1/Hz)','Interpreter','latex');
xlim([0,fs/2]);
grid minor;

subplot(4,1,4);
plot(f,PSD_2_C4);
title(['PSD of C4 signal at 2700m height - trial number ', num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel('Frequency (Hz)','Interpreter','latex');
ylabel('PSD (1/Hz)','Interpreter','latex');
xlim([0,fs/2]);
grid minor;

%% Part e and f
clc; close all;

[~ , ~, ~, ps_1_C3] = spectrogram(EEG_1_C3(trial_slct,:), hamming(32), 16, 256, fs, "psd");
[~ , ~, ~, ps_2_C3] = spectrogram(EEG_2_C3(trial_slct,:), hamming(32), 16, 256, fs, "psd");
[~ , ~, ~, ps_1_C4] = spectrogram(EEG_1_C4(trial_slct,:), hamming(32), 16, 256, fs, "psd");
[~ , f, t, ps_2_C4] = spectrogram(EEG_2_C4(trial_slct,:), hamming(32), 16, 256, fs, "psd");

E_alpha_1_C3 = sum(ps_1_C3(9:14, :));
E_alpha_2_C3 = sum(ps_2_C3(9:14, :));
E_alpha_1_C4 = sum(ps_1_C4(9:14, :));
E_alpha_2_C4 = sum(ps_2_C4(9:14, :));

E_beta_1_C3 = sum(ps_1_C3(15:19, :));
E_beta_2_C3 = sum(ps_2_C3(15:19, :));
E_beta_1_C4 = sum(ps_1_C4(15:19, :));
E_beta_2_C4 = sum(ps_2_C4(15:19, :));

figure;
subplot(4,2,1);
plot(t,E_alpha_1_C3);
title(['Energy of alpha band - C3 - 990m - trial number ', num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Energy','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,2);
plot(t,E_alpha_2_C3);
title(['Energy of alpha band - C3 - 2700m - trial number ', num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Energy','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,3);
plot(t,E_alpha_1_C4);
title(['Energy of alpha band - C4 - 990m - trial number ', num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Energy','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,4);
plot(t,E_alpha_2_C4);
title(['Energy of alpha band - C4 - 2700m - trial number ', num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Energy','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,5);
plot(t,E_beta_1_C3);
title(['Energy of beta band - C3 - 990m - trial number ', num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Energy','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,6);
plot(t,E_beta_2_C3);
title(['Energy of beta band - C3 - 2700m - trial number ', num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Energy','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,7);
plot(t,E_beta_1_C4);
title(['Energy of beta band - C4 - 990m - trial number ', num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Energy','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,8);
plot(t,E_beta_2_C4);
title(['Energy of beta band - C4 - 2700m - trial number ', num2str(trial_slct)],'Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Energy','Interpreter','latex');
xlim([0,6]);
grid minor;

%% Q1 - Part g
clc;
E_alpha_1_C3_mean = zeros(1,95);
E_alpha_2_C3_mean = zeros(1,95);
E_alpha_1_C4_mean = zeros(1,95);
E_alpha_2_C4_mean = zeros(1,95);

E_beta_1_C3_mean = zeros(1,95);
E_beta_2_C3_mean = zeros(1,95);
E_beta_1_C4_mean = zeros(1,95);
E_beta_2_C4_mean = zeros(1,95);

for i = 1:37
    [~ , ~, ~, ps_1_C3] = spectrogram(EEG_1_C3(i,:), hamming(32), 16, 256, fs, "psd");
    [~ , f, t, ps_1_C4] = spectrogram(EEG_1_C4(i,:), hamming(32), 16, 256, fs, "psd");

    E_alpha_1_C3_mean = E_alpha_1_C3_mean + sum(ps_1_C3(9:14, :));
    E_alpha_1_C4_mean = E_alpha_1_C4_mean + sum(ps_1_C4(9:14, :));
    E_beta_1_C3_mean = E_beta_1_C3_mean + sum(ps_1_C3(15:19, :));
    E_beta_1_C4_mean = E_beta_1_C4_mean + sum(ps_1_C4(15:19, :));
end

E_alpha_1_C3_mean = E_alpha_1_C3_mean / 37;
E_alpha_1_C4_mean = E_alpha_1_C4_mean / 37;
E_beta_1_C3_mean = E_beta_1_C3_mean / 37;
E_beta_1_C4_mean = E_beta_1_C4_mean / 37;


for i = 1:41
    [~ , ~, ~, ps_2_C3] = spectrogram(EEG_2_C3(i,:), hamming(32), 16, 256, fs, "psd");
    [~ , f, t, ps_2_C4] = spectrogram(EEG_2_C4(i,:), hamming(32), 16, 256, fs, "psd");
    
    E_alpha_2_C3_mean = E_alpha_2_C3_mean + sum(ps_2_C3(9:14, :));
    E_alpha_2_C4_mean = E_alpha_2_C4_mean + sum(ps_2_C4(9:14, :));
    E_beta_2_C3_mean = E_beta_2_C3_mean + sum(ps_2_C3(15:19, :));
    E_beta_2_C4_mean = E_beta_2_C4_mean + sum(ps_2_C4(15:19, :));
end

E_alpha_2_C3_mean = E_alpha_2_C3_mean / 41;
E_alpha_2_C4_mean = E_alpha_2_C4_mean / 41;
E_beta_2_C3_mean = E_beta_2_C3_mean / 41;
E_beta_2_C4_mean = E_beta_2_C4_mean / 41;

figure;
subplot(4,2,1);
plot(t,E_alpha_1_C3_mean);
title('Energy of alpha band - C3 - 990m - mean of all trials ','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Energy','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,2);
plot(t,E_alpha_2_C3_mean);
title('Energy of alpha band - C3 - 2700m - mean of all trials ','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Energy','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,3);
plot(t,E_alpha_1_C4_mean);
title('Energy of alpha band - C4 - 990m - mean of all trials ','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Energy','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,4);
plot(t,E_alpha_2_C4_mean);
title('Energy of alpha band - C4 - 2700m - mean of all trials ','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Energy','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,5);
plot(t,E_beta_1_C3_mean);
title('Energy of beta band - C3 - 990m - mean of all trials ','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Energy','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,6);
plot(t,E_beta_2_C3_mean);
title('Energy of beta band - C3 - 2700m - mean of all trials ','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Energy','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,7);
plot(t,E_beta_1_C4_mean);
title('Energy of beta band - C4 - 990m - mean of all trials ','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Energy','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,8);
plot(t,E_beta_2_C4_mean);
title('Energy of beta band - C4 - 2700m - mean of all trials ','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Energy','Interpreter','latex');
xlim([0,6]);
grid minor;

%% Q1 - Part h
clc;

E_alpha_1_C3_ref = sum(E_alpha_1_C3_mean(8:24)) / 17;
E_alpha_1_C4_ref = sum(E_alpha_1_C4_mean(8:24)) / 17;
E_beta_1_C3_ref = sum(E_beta_1_C3_mean(8:24)) / 17;
E_beta_1_C4_ref = sum(E_beta_1_C4_mean(8:24)) / 17;
E_alpha_2_C3_ref = sum(E_alpha_2_C3_mean(8:24)) / 17;
E_alpha_2_C4_ref = sum(E_alpha_2_C4_mean(8:24)) / 17;
E_beta_2_C3_ref = sum(E_beta_2_C3_mean(8:24)) / 17;
E_beta_2_C4_ref = sum(E_beta_2_C4_mean(8:24)) / 17;

E_alpha_1_C3_diff = (E_alpha_1_C3_mean - E_alpha_1_C3_ref)/E_alpha_1_C3_ref * 100;
E_alpha_1_C4_diff = (E_alpha_1_C4_mean - E_alpha_1_C4_ref)/E_alpha_1_C4_ref * 100;
E_beta_1_C3_diff = (E_beta_1_C3_mean - E_beta_1_C3_ref)/E_beta_1_C3_ref * 100;
E_beta_1_C4_diff = (E_beta_1_C4_mean - E_beta_1_C4_ref)/E_beta_1_C4_ref * 100;
E_alpha_2_C3_diff = (E_alpha_2_C3_mean - E_alpha_2_C3_ref)/E_alpha_2_C3_ref * 100;
E_alpha_2_C4_diff = (E_alpha_2_C4_mean - E_alpha_2_C4_ref)/E_alpha_2_C4_ref * 100;
E_beta_2_C3_diff = (E_beta_2_C3_mean - E_beta_2_C3_ref)/E_beta_2_C3_ref * 100;
E_beta_2_C4_diff = (E_beta_2_C4_mean - E_beta_2_C4_ref)/E_beta_2_C4_ref * 100;

figure;
subplot(4,2,1);
plot(t,E_alpha_1_C3_diff);
title('Percentage of changes of alpha band energy - C3 - 990m - mean of all trials ','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Percentage of changes','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,2);
plot(t,E_alpha_2_C3_diff);
title('Percentage of changes of alpha band energy - C3 - 2700m - mean of all trials ','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Percentage of changes','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,3);
plot(t,E_alpha_1_C4_diff);
title('Percentage of changes of alpha band energy - C4 - 990m - mean of all trials ','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Percentage of changes','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,4);
plot(t,E_alpha_2_C4_diff);
title('Percentage of changes of alpha band energy - C4 - 2700m - mean of all trials ','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Percentage of changes','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,5);
plot(t,E_beta_1_C3_diff);
title('Percentage of changes of beta band energy - C3 - 990m - mean of all trials ','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Percentage of changes','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,6);
plot(t,E_beta_2_C3_diff);
title('Percentage of changes of beta band energy - C3 - 2700m - mean of all trials ','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Percentage of changes','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,7);
plot(t,E_beta_1_C4_diff);
title('Percentage of changes of beta band energy - C4 - 990m - mean of all trials ','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Percentage of changes','Interpreter','latex');
xlim([0,6]);
grid minor;

subplot(4,2,8);
plot(t,E_beta_2_C4_diff);
title('Percentage of changes of beta band energy - C4 - 2700m - mean of all trials ','Interpreter','latex','FontSize',14);
xlabel('Time (s)','Interpreter','latex');
ylabel('Percentage of changes','Interpreter','latex');
xlim([0,6]);
grid minor;

%% Question 2
clc; clear; close all;

%% Q2 - Part a
L = 8000; % Length of signal
noise_rand = -1 + 2*rand(1,L); 

figure;
subplot(3,1,1);
plot(noise_rand);
title("Noise signal", 'Interpreter','latex','FontSize',14);
subplot(3,1,2);
histogram(noise_rand,100);
title("Histogram of noise", 'Interpreter','latex','FontSize',14);
subplot(3,1,3);
periodogram(noise_rand);

%% Q2 - Part b
clc; close all;
mu = 0; % Mean of signal
sigma = 1; % STD of signal
noise_randn = sigma*randn(1,L) + mu;

figure;
subplot(3,1,1);
plot(noise_randn);
title("Noise signal", 'Interpreter','latex','FontSize',14);
subplot(3,1,2);
histogram(noise_randn,100);
title("Histogram of noise", 'Interpreter','latex','FontSize',14);
subplot(3,1,3);
periodogram(noise_randn);

%% Q2 - Part c
clc; close all;

ab = rand(2,L);
mu = 0;
sigma = 1;
noise_x = sigma*cos(2*pi*ab(2,:)).*sqrt(-2*log(1-ab(1,:)))+mu;

figure;
subplot(3,1,1);
plot(noise_x);
title("Noise signal", 'Interpreter','latex','FontSize',14);
subplot(3,1,2);
histogram(noise_x,100);
title("Histogram of noise", 'Interpreter','latex','FontSize',14);
subplot(3,1,3);
periodogram(noise_x);
%% Q2 - Part d
clc; close all;

noise_wgn = wgn(1,L,1);

figure;
subplot(3,1,1);
plot(noise_wgn);
title("Noise signal", 'Interpreter','latex','FontSize',14);
subplot(3,1,2);
histogram(noise_wgn,100);
title("Histogram of noise", 'Interpreter','latex','FontSize',14);
subplot(3,1,3);
periodogram(noise_wgn);

%% Q2 - Part e
clc; close all;

[corr_noise_rand, ~] = xcorr(noise_rand);
[corr_noise_randn, ~] = xcorr(noise_randn);
[corr_noise_x, lags] = xcorr(noise_x);

NFFT = 2^nextpow2(length(corr_noise_rand));

fft_corr_rand = fft(corr_noise_rand,NFFT);
fft_corr_randn = fft(corr_noise_randn,NFFT);
fft_corr_x = fft(corr_noise_x,NFFT);

fs = 1000;
f = fs/NFFT*(-NFFT/2:NFFT/2-1); % frequency axes

figure;
subplot(3,2,1);
plot(f,abs(fftshift(fft_corr_rand)));
title("FT of auto correlation of first noise signal", 'Interpreter','latex','FontSize',14);

subplot(3,2,2);
plot(lags, corr_noise_rand);
title("Auto correlation of first noise signal","Interpreter","latex","Fontsize",14);

subplot(3,2,3);
plot(f,abs(fftshift(fft_corr_randn)));
title("FT of auto correlation of second noise signal", 'Interpreter','latex','FontSize',14);

subplot(3,2,4);
plot(lags, corr_noise_randn);
title("Auto correlation of second noise signal","Interpreter","latex","Fontsize",14);

subplot(3,2,5);
plot(f,abs(fftshift(fft_corr_x)));
title("FT of auto correlation of third noise signal", 'Interpreter','latex','FontSize',14);

subplot(3,2,6);
plot(lags, corr_noise_x);
title("Auto correlation of third noise signal","Interpreter","latex","Fontsize",14);

%% Question 3
%% Q3 - Part a
clc; clear;
sigma_x = 1;
a = 0.6;
m_period = 0:100;

Ry = sigma_x / (1-a^2) * (- a).^abs(m_period);

figure;
plot(m_period, Ry);
title("$$R_Y[m]$$","Interpreter","latex","FontSize",14);
xlabel("m","Interpreter","latex","FontSize",14);
ylabel("$$R_Y$$","Interpreter","latex","FontSize",14);

%% Q3 - Part b
clc;
x = randn(1,10000);
n = 0:10000-1;
y = filter(1, [1 a], x);

figure;
subplot(2,1,1);
plot(n,x);
title("input white noise signal = x[n]","Interpreter","latex","FontSize",14);
xlabel("n","Interpreter","latex","FontSize",14);
ylabel("Amplitude","Interpreter","latex","FontSize",14);

subplot(2,1,2);
plot(n,y);
title("output signal = y[n]","Interpreter","latex","FontSize",14);
xlabel("n","Interpreter","latex","FontSize",14);
ylabel("Amplitude","Interpreter","latex","FontSize",14);

%% Q3 - Part c
clc;

m_period = 0:100;
figure;
counter = 1;
for N = [100 500 1000 5000]
    N_sample = 1:N;
    [Ry ,Ry_biased, Ry_unbiased] = my_corr(y, N_sample, m_period);
    subplot(4,1,counter);
    plot(m_period,Ry,"LineWidth",1);
    hold on;
    plot(m_period,Ry_biased,"LineWidth",1);
    hold on;
    plot(m_period,Ry_unbiased,"LineWidth",1);
    title(['N = ',num2str(N),' - samples 1:',num2str(N)],"Interpreter","latex","FontSize",14);
    xlabel("m","Interpreter","latex");
    ylabel("Amplitude","Interpreter","latex");
    legend(["$$R_y$$","$$\hat{R_y}$$ biased","$$\hat{R_y}$$ unbiased"],'Interpreter','latex');
    counter = counter+1;
end

counter = 1;
figure;
for start_point = [1, 3000, 6000, 9501]
    N = 500;
    N_sample = start_point:start_point+N-1;
    [Ry ,Ry_biased, Ry_unbiased] = my_corr(y, N_sample, m_period);
    
    subplot(4,1,counter);
    plot(m_period,Ry,"LineWidth",1);
    hold on;
    plot(m_period,Ry_biased,"LineWidth",1);
    hold on;
    plot(m_period,Ry_unbiased,"LineWidth",1);
    title(['N = 500 - samples ',num2str(start_point),':500'],"Interpreter","latex","FontSize",14);
    xlabel("m","Interpreter","latex");
    ylabel("Amplitude","Interpreter","latex");
    legend(["$$R_y$$","$$\hat{R_y}$$ biased","$$\hat{R_y}$$ unbiased"],'Interpreter','latex');
    counter = counter+1;
end

%% Q3 - Part d
close all; clc;

figure;
counter = 1;
for M = [2 5 10 20]
    disp(["==================== M = ",num2str(M)," =========================="]);
    R = AC_matrix(M);
    if(isequal(R, R.'))
        disp("Auto Corrolation Matrix Is Symmetric")
    else
        disp("Auto Corrolation Matrix Isn't Symmetric")
    end
    disp("Determinant of Auto Correlation Matrix:");
    disp(det(R));
    disp("Eigenvalues of Auto Correlation Matrix:");
    disp(eig(R));
    disp("Diagonal Elements of Auto Correlation Matrix:")
    disp(diag(R));
    
    subplot(2,2,counter);
    imagesc(R);
    colorbar;
    xlabel("Column","Interpreter","latex");
    ylabel("Row","Interpreter","latex");
    title(['Autocorrelation Matrix for M =',num2str(M)],"Interpreter","latex","FontSize",14);
    disp("======================================================");
    counter = counter+1;
end

%% Q3 - Part e
clc; close all;

figure;
counter = 1;
for M = [2 5 10 20]
    disp(["==================== M = ",num2str(M)," =========================="]);
    R_estimated = AC_matrix_est(M,y, N_sample, m_period);
    if(isequal(R_estimated, R_estimated.'))
        disp("Biased Estimated Auto Corrolation Matrix Is Symmetric")
    else
        disp("Biased Estimated Auto Corrolation Matrix Isn't Symmetric")
    end
    disp("Determinant of Estimated Auto Correlation Matrix:");
    disp(det(R_estimated));
    disp("Eigenvalues of Estimated Auto Correlation Matrix:");
    disp(eig(R_estimated));
    disp("Diagonal Elements of Estimated Auto Correlation Matrix:")
    disp(diag(R_estimated));
    
    subplot(2,2,counter);
    imagesc(R_estimated);
    colorbar;
    xlabel("Column","Interpreter","latex");
    ylabel("Row","Interpreter","latex");
    title(['Estimated Autocorrelation Matrix for M =',num2str(M)],"Interpreter","latex","FontSize",14);
    disp("======================================================");
    counter = counter+1;
end

%% Q3 - Part f
clc; close all;

load mit200

m_period = 0:5000;
N_sample = 1:5000;
[~ ,Ry_biased, ~] = my_corr(ecgsig, N_sample, m_period);

figure;
subplot(3,2,[1,2]);
plot(m_period,Ry_biased,"LineWidth",1);
title("Estmated Auto Correlation of ECG signal","Interpreter","latex","FontSize",14);
xlabel("m","Interpreter","latex");
ylabel("$$\hat{R}_{ECG}[m]$$","Interpreter","latex");

counter = 3;
for M = [2 5 10 20]
    R_estimated = AC_matrix_est(M,ecgsig, N_sample, m_period);
    subplot(3,2,counter);
    imagesc(R_estimated);
    colorbar;
    xlabel("Column","Interpreter","latex");
    ylabel("Row","Interpreter","latex");
    title(['Estimated Autocorrelation Matrix for M =',num2str(M)],"Interpreter","latex","FontSize",14);
    counter = counter+1;
end
%% Q3 - Part g
clc; close all;
EEG = load('EEG_2700mS16.mat').y(1,:,1);

m_period = 0:1000;
N_sample = 1:1000;
[~ ,Ry_biased, ~] = my_corr(EEG, N_sample, m_period);

figure;
subplot(3,2,[1,2]);
plot(m_period,Ry_biased,"LineWidth",1);
title("Estmated Auto Correlation of EEG signal","Interpreter","latex","FontSize",14);
xlabel("m","Interpreter","latex");
ylabel("$$\hat{R}_{EEG}[m]$$","Interpreter","latex");

counter = 3;
for M = [2 5 10 20]
    R_estimated = AC_matrix_est(M,EEG, N_sample, m_period);
    subplot(3,2,counter);
    imagesc(R_estimated);
    colorbar;
    xlabel("Column","Interpreter","latex");
    ylabel("Row","Interpreter","latex");
    title(['Estimated Autocorrelation Matrix for M =',num2str(M)],"Interpreter","latex","FontSize",14);
    counter = counter+1;
end
%% Question 4
clc; clear; close all;

fs = 2048;
window_size = 1; % 1 second
su3_trials = [];
su6_trials = [];

files = dir('subject3/*.mat');
for i=1:24
    data = load(['subject3/',files(i).name]);
    su3_trials = [su3_trials; seperate_trials(data, window_size, fs)];
end

files = dir('subject7/*.mat');
for i=1:24
    data = load(['subject7/',files(i).name]);
    su6_trials = [su6_trials; seperate_trials(data, window_size, fs)];
end

P300_su3 = mean(su3_trials,1);
P300_su6 = mean(su6_trials,1);

% fs = 32;
t = (0:1/fs:window_size - 1/fs)*1000;

figure;
subplot(2,1,1);
plot(t, P300_su3);
title('P300 Waveform on Pz channel - Subject 3','Interpreter','latex','FontSize',14);
xlabel("Time (ms)",'Interpreter','latex','FontSize',14);

subplot(2,1,2);
plot(t, P300_su6);
title('P300 Waveform on Pz channel - Subject 7','Interpreter','latex','FontSize',14);
xlabel("Time (ms)",'Interpreter','latex','FontSize',14);

%% Question 5
clc; clear; close all;

data = load("PPG_a44542m.mat");
fs = 125;
respiratory_signal = data.val(1,:);
PPG_signal = data.val(2,:);
ECG_lead_1 = data.val(3,:);
ECG_lead_2 = data.val(4,:);
ECG_lead_3 = data.val(5,:);

%% Q5 - Part a
t = 0:1/fs:10-1/fs;

figure;
subplot(5,1,1);
plot(t, respiratory_signal(1:fs*10));
title('Respiratory signal','Interpreter','latex','FontSize',14);
xlabel("Time (s)","Interpreter","latex");


subplot(5,1,2);
plot(t, PPG_signal(1:fs*10));
title('PPG signal','Interpreter','latex','FontSize',14);
xlabel("Time (s)","Interpreter","latex");

subplot(5,1,3);
plot(t, ECG_lead_1(1:fs*10));
title('ECG signal - Lead 1','Interpreter','latex','FontSize',14);
xlabel("Time (s)","Interpreter","latex");


subplot(5,1,4);
plot(t, ECG_lead_2(1:fs*10));
title('ECG signal - Lead 2','Interpreter','latex','FontSize',14);
xlabel("Time (s)","Interpreter","latex");

subplot(5,1,5);
plot(t, ECG_lead_3(1:fs*10));
title('ECG signal - Lead 3','Interpreter','latex','FontSize',14);
xlabel("Time (s)","Interpreter","latex");


%% Q5 - Part b
clc; close all;
P = 30;
L = length(PPG_signal);
omega_0 = zeros(1, L);
omega_2 = zeros(1, L);
omega_4 = zeros(1, L);
for n = P:L
    for k = (n-P+1):n
        omega_0(n) = omega_0(n) + (deriv(PPG_signal, k, 0))^2;
        omega_2(n) = omega_2(n) + (deriv(PPG_signal, k, 1))^2;
        omega_4(n) = omega_4(n) + (deriv(PPG_signal, k, 2))^2;
    end
    omega_0(n) = 2*pi/P * omega_0(n);
    omega_2(n) = 2*pi/P * omega_2(n);
    omega_4(n) = 2*pi/P * omega_4(n);
end

figure;
subplot(3,1,1);
plot(P+1:L-1,omega_0(1,P+1:end-1));
title("Estimation of 0th-moment of PPG signal - P = 20","Interpreter","latex","Fontsize",14);
ylabel("$$\hat{\omega_0}[n]$$","Interpreter","latex","Fontsize",14);
xlabel("n","Interpreter","latex","Fontsize",14);

subplot(3,1,2);
plot(P+1:L-1,omega_2(P+1:end-1));
title("Estimation of 2th-moment of PPG signal - P = 20","Interpreter","latex","Fontsize",14);
ylabel("$$\hat{\omega_2}[n]$$","Interpreter","latex","Fontsize",14);
xlabel("n","Interpreter","latex","Fontsize",14);

subplot(3,1,3);
plot(P+1:L-1,omega_4(P+1:end-1));
title("Estimation of 4th-moment of PPG signal - P = 20","Interpreter","latex","Fontsize",14);
ylabel("$$\hat{\omega_4}[n]$$","Interpreter","latex","Fontsize",14);
xlabel("n","Interpreter","latex","Fontsize",14);




%% Q5 - Part c
clc; close all;
H0 = omega_0;
H1 = sqrt(omega_2./omega_0);
H2 = sqrt(omega_4./omega_2 - omega_2./omega_0);

figure;
subplot(3,1,1);
plot(P+1:L-1,H0(1,P+1:end-1));
title("Hjorth Parameters - Activity","Interpreter","latex","Fontsize",14);
ylabel("$$H_0[n]$$","Interpreter","latex","Fontsize",14);
xlabel("n","Interpreter","latex","Fontsize",14);

subplot(3,1,2);
plot(P+1:L-1,H1(1,P+1:end-1));
title("Hjorth Parameters - Mobility","Interpreter","latex","Fontsize",14);
ylabel("$$H_1[n]$$","Interpreter","latex","Fontsize",14);
xlabel("n","Interpreter","latex","Fontsize",14);

subplot(3,1,3);
plot(P+1:L-1,H2(1,P+1:end-1));
title("Complexity","Interpreter","latex","Fontsize",14);
ylabel("$$H_2[n]$$","Interpreter","latex","Fontsize",14);
xlabel("n","Interpreter","latex","Fontsize",14);
%% Functions
function result = seperate_trials(data, window_size, fs)
    EEG = data.data(13, :);
%     [b,a] = butter(6,[0.5 12]/(fs/2));
%     EEG = filtfilt(b,a,EEG);
%     EEG = downsample(EEG, 64);
%     fs = 32;
    p = prctile(EEG,[10 90]);
    i1 = EEG < p(1); 
    v1 = min(EEG(~i1));
    i2 = EEG > p(2); 
    v2 = max(EEG(~i2));
    EEG(i1) = v1;
    EEG(i2) = v2;
    trials_num = size(data.stimuli,2);
    target = data.target;
    stim = data.stimuli;
    result = [];
    for i = 1:trials_num
        if (stim(i) == target)
            start_sample = floor(i*0.4*fs);
            end_sample = start_sample + floor(window_size*fs) - 1;
            result = [result; EEG(1,start_sample:end_sample)];
        end
    end
end

function [corr_actual, corr_biased, corr_unbiased] = my_corr(signal, N_sample, m_period)
    N = length(N_sample);
    if (N <= length(signal))
        signal_sampled = signal(N_sample);
    else
        signal_sampled = signal;
    end

    corr_biased = zeros(1,length(m_period));
    corr_unbiased = zeros(1, length(m_period));
    
    for m = m_period
        for j = 1:N-m
            corr_biased(m+1) = corr_biased(m+1) + signal_sampled(j)*signal_sampled(j+m);
        end
        corr_unbiased(m+1) = corr_biased(m+1) / (N-m);
        corr_biased(m+1) = corr_biased(m+1) / N;
    end

    corr_actual = 1 / (1-0.6^2) * (- 0.6).^abs(m_period);

end

function matrix = AC_matrix(N)

    matrix = zeros(N);
    for i=1:N
        for j = 1:N
            matrix(i,j) =  4/3 * (-0.5).^abs(i-j);
        end
    end
end

function matrix_estimated = AC_matrix_est(N,signal, N_sample, m_period)
    [~, corr_biased, ~] = my_corr(signal, N_sample, m_period);
    matrix_estimated = zeros(N);
    for i=1:N
        for j = 1:N
            matrix_estimated(i,j) =  corr_biased(abs(i-j)+1);
        end
    end

end


function result = deriv(x, n, order)
    
    if(order == 0)
        result = x(n);
    elseif(order == 1 && n ~= 1)
        result = x(n) - x(n-1);
    elseif(order == 1 && n==1)
        result = x(n);
    elseif(order == 2 && n~= 1 && n~=length(x))
        result = (x(n+1) - x(n))-(x(n)-x(n-1));
    elseif(order == 2 && n==1)
        result = (x(n+1) - x(n)) - x(n);
    elseif(order == 2 && n==length(x))
        result = -x(n)-(x(n)-x(n-1));
    end
end












