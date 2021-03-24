function B = doNMF(trainfcs,K,niter,Binit,Winit)

F = size(trainfcs,1); 
T = size(trainfcs,2);

W = Winit;
B = Binit;

inds = setdiff(1:sum(K),[]);
ONES = ones(F,T);

for i=1:niter
    
    % Updating the activations.
    W = W .* (B'*( trainfcs./(B*W+eps))) ./ (B'*ONES);
    
    % Updating the basis vectors.
    B(:,inds) = B(:,inds) .* ((trainfcs./(B*W+eps))*W(inds,:)') ./(ONES*W(inds,:)');
end
sumB = sum(B);
B = B*diag(1./sumB);
end