function [speech_recv, music_recv] = separate_signals(mixed_spec,Bmusic,Bspeech,niter,K)

F = size(mixed_spec,1); 
T = size(mixed_spec,2);

W = 1+rand(sum(2*K), T);
B = cat(2,Bspeech,Bmusic);

ONES = ones(F,T);

for i=1:niter
    % Updating the activations.
    W = W .* (B'*( mixed_spec./(B*W+eps))) ./ (B'*ONES);
end

Wspeech = W(1:K,:);
Wmusic = W(K+1:end,:);

speech_recv = Bspeech*Wspeech;
music_recv = Bmusic*Wmusic;

end