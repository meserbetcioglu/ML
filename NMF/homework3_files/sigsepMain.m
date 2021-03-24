clear, clc, close all;

musicw = audioread(fullfile('musicf1.wav'));
speechw = audioread(fullfile('speechf1.wav'));
mixedw = audioread(fullfile('mixedf1.wav'));

% Short-time Fourier Transform of music and speech signals. Gives us
% spectrogram of the signals.
music_spec = stft(musicw',2048,256,0,hann(2048));
speech_spec = stft(speechw',2048,256,0,hann(2048));

% Magnitude and phase analysis of the signals
music_mag = abs(music_spec);
music_phase = music_spec ./(abs(music_spec)+eps);

speech_mag = abs(speech_spec);
speech_phase = speech_spec ./(abs(speech_spec)+eps);

% Spectrogram, magnitude and phase of mixed signals.
mixed_spec = stft(mixedw',2048,256,0,hann(2048));
mixed_mag = abs(mixed_spec);
mixed_phase = mixed_spec ./(abs(mixed_spec)+eps);

% Conditions for NMF. K is the number of basis vectors, niter is total
% iteration of learning.
K = 200;
niter = 250;

% Initial basis vectors and activations.
load(fullfile('Bminit.mat'));
load(fullfile('Wminit.mat'));

load(fullfile('Bsinit.mat'));
load(fullfile('Wsinit.mat'));

% NMF of signals, gives learned basis vectors.
Bmusic = doNMF(music_spec,K,niter,Bm,Wm);
Bspeech = doNMF(speech_spec,K,niter,Bs,Ws);

% Seperation of speech and music signals from the mixed signal. Almost same
% as NMF but with each iteration only activations get updated and
% previously learned basis vectors are used.
[speech_recv, music_recv] = separate_signals(mixed_spec,Bmusic,Bspeech, niter,K);

% Reconstruction of signals in time.
speech_rec = stft(mixed_phase.*speech_recv, 2048, 256, 0, hann(2048));
speech_rec = speech_rec';
music_rec = stft(mixed_phase.*music_recv, 2048, 256, 0, hann(2048));
music_rec = music_rec';

% Audio recording of recreated speech and music data with 16KHz sampling
% frequency.
Fs = 16000;
speechname = 'speech_recreated.wav';
musicname = 'music_recreated.wav';
audiowrite(speechname,speech_rec,Fs);
audiowrite(musicname,music_rec,Fs);

% Learned basis vectors are saved at Bs.mat and Bm.mat files.
save('Bs.mat', 'Bspeech');
save('Bm.mat', 'Bmusic');



