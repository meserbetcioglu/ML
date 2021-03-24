clear, clc, close all;

% fea -> Each row represents a face
% gnd -> Each row represents the classification of a face.
% gnd(i) = -1 => fea(i,:) is neutral
% gnd(i) = -2 => fea(i,:) is smiling
fileDir = fullfile(pwd, 'face_databases/ORL_32x32.mat');
load(fileDir)

w = 32; 
h = 32; 
numFaces = 400;
% % First 40 will be put into the
% % training set.
% for i=1:40
%     face(1:32,1:32) = reshape(fea(i,:),[h,w]);
%     figure
%     imagesc(face);colormap(gray);
%     title(i)
% end

% Determining the size of neutral and smile classes.
neutralSize = 0;
smileSize = 0;
for i=1:40
    if gnd(i) == -1
        neutralSize = neutralSize + 1;
    elseif gnd(i) == -2
        smileSize = smileSize + 1;
    end
end
neutral = zeros(1024, neutralSize);
smile = zeros(1024, smileSize);

% Each column is a face instead of each row.
faces = reshape(fea,1024,400);

% neutral and smile classes are populated.
neutralIndex = 1;
smileIndex = 1;
for i=1:40
    if gnd(i) == -1
        neutral(:,neutralIndex) = faces(:,i);
        neutralIndex = neutralIndex + 1;
    elseif gnd(i) == -2
        smile(:,smileIndex) = faces(:,i);
        smileIndex = smileIndex + 1;
    end
end

% The mean of each faces is subtracted.
mean_1 = mean(faces, 2);
mean_2 = repmat(mean_1, 1, size(faces,2));
faces = faces - mean_2;

% Singular value decomposition, which gives eigenvectors and eigenvalues.
[u,s,v] = svd(faces, 0);
eigVals = diag(s);
eigVecs = u;

figure; imagesc(reshape(mean_1, h, w)); title('Mean Face'); colormap(gray);
figure;
subplot(2, 2, 1); imagesc(reshape(u(:, 1), h, w)); colormap(gray); title('First Eigenface');
subplot(2, 2, 2); imagesc(reshape(u(:, 2), h, w)); colormap(gray); title('Second Eigenface');
subplot(2, 2, 3); imagesc(reshape(u(:, 3), h, w)); colormap(gray); title('Third Eigenface');
subplot(2, 2, 4); imagesc(reshape(u(:, 4), h, w)); colormap(gray); title('Fourth Eigenface'); 

% The mean of smiling and neutral faces are subtracted.
neutral = getMean(neutral);
smile = getMean(smile);

% Svd of neutral faces.
[uN,sN,vN] = svd(neutral, 0);
eigValsN = diag(sN);
eigVecsN = uN;

% Svd of smiling faces.
[uS,sS,vS] = svd(smile, 0);
eigValsS = diag(sS);
eigVecsS = uS;

% Energy of each neutral face eigenvector is calculated to select high
% energy eigenvectors.
energyN = zeros(size(uN,2),1);
for i = 1:size(uN,2)
energyN(i) = sum(eigValsN(1:i));
end
propEnergyN = energyN./energyN(end);
percentMarkN = min(find(propEnergyN > 0.9));
eigenVecsN = uN(:, 1:percentMarkN);

% Energy of each smiling face eigenvector is calculated to select high
% energy eigenvectors.
energyS = zeros(size(uS,2),1);
for i = 1:size(uS,2)
energyS(i) = sum(eigValsS(1:i));
end
propEnergyS = energyS./energyS(end);
percentMarkS = min(find(propEnergyS > 0.9));
eigenVecsS = uS(:, 1:percentMarkS);

% Each eigen vector is normalized.
for i = 1:size(eigenVecsN,2)
eigenVecsN(:,i) = eigenVecsN(:,i)./sqrt(sum(eigenVecsN(:,i).^2));
end
for i = 1:size(eigenVecsS,2)
eigenVecsS(:,i) = eigenVecsS(:,i)./sqrt(sum(eigenVecsS(:,i).^2));
end

% Projection of all faces onto neutral and smiling eigenvectors stored as
% weights.
nWeights = pinv(eigenVecsN)*faces;
sWeights = pinv(eigenVecsS)*faces;

% For faces other than in training set, a weight difference is calculated
% by subtracting the euclidian norm of each weight. If the
% difference is greater than zero, picture is predicted to be of a
% neutral face. Else, picture is predicted to be of a smiling face.
smileCount = 0;
neutralCount = 0;
gndPred = string.empty;
for i = 1:numFaces
    Wdif = sqrt(sum(nWeights(:,i).^2)) - sqrt(sum(sWeights(:,i).^2));
    if Wdif > 0
        gndPred(i) = 'neutral';
        neutralCount = neutralCount + 1;
    else
        gndPred(i) = 'smile';
        smileCount = smileCount + 1;
    end
end
gndPred = gndPred';

yesNcount = 0;
noNcount = 0;
yesScount = 0;
noScount = 0;
for i = 1:400
    if gndPred(i) == 'smile'
        if gnd(i) == -2
            yesScount = yesScount + 1;
        else
            noScount = noScount + 1;
        end
    else
        if gnd(i) == -1
            yesNcount = yesNcount + 1;
        else
            noNcount = noNcount + 1;
        end
    end       
end
fprintf("Predicted Neutral right: %d \t\t Predicted Neutral but were wrong: %d \n",yesNcount, noNcount)

fprintf("Predicted Smiling right: %d \t\t Predicted Smiling but were wrong: %d \n",yesScount, noScount)

save(fileDir, 'fea', 'gnd', 'gndPred')

function meanMat = getMean(Mat)
mean_1 = mean(Mat, 2);
mean_2 = repmat(mean_1, 1, size(Mat,2));
meanMat = Mat - mean_2;
end
