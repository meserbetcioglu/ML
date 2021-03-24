clear, clc, close all;

w = 36; 
h = 36; 
trainFaces = 2500;
testFaces = 200;

training_fileDir = fullfile(pwd, 'gender_classification/training');
testing_fileDir = fullfile(pwd, 'gender_classification/testing');

for k = 1:trainFaces
  trainMenFilename = strcat(training_fileDir,'\men\',num2str(k), '.jpg');
  imageData = imread(trainMenFilename);
  trainMen_int(:,k) = reshape(imageData,[],1);
end
trainMen = im2double(trainMen_int);

for k = 1:trainFaces
  trainWomenFilename = strcat(training_fileDir,'\women\',num2str(k), '.jpg');
  imageData = imread(trainWomenFilename);
  trainWomen_int(:,k) = reshape(imageData,[],1);
end
trainWomen = im2double(trainWomen_int);

for k = 1:testFaces
  testMenFilename = strcat(testing_fileDir,'\men\',num2str(k), '.jpg');
  imageData = imread(testMenFilename);
  testMen_int(:,k) = reshape(imageData,[],1);
end
testMen = im2double(testMen_int);


for k = 1:testFaces
  testWomenFilename = strcat(testing_fileDir,'\women\',num2str(k), '.jpg');
  imageData = imread(testWomenFilename);
  testWomen_int(:,k) = reshape(imageData,[],1);
end
testWomen = im2double(testWomen_int);

trainMen = getMean(trainMen);
trainWomen = getMean(trainWomen);

% Svd of male faces.
[uM,sM,vM] = svd(trainMen, 0);
eigValsM = diag(sM);
eigVecsM = uM;

% Svd of female faces.
[uF,sF,vF] = svd(trainWomen, 0);
eigValsF = diag(sF);
eigVecsF = uF;

% Energy of each neutral face eigenvector is calculated to select high
% energy eigenvectors.
energyM = zeros(size(uM,2),1);
energyThM = 0.999;
for i = 1:size(uM,2)
energyM(i) = sum(eigValsM(1:i));
end
propEnergyM = energyM./energyM(end);
percentMarkM = size(eigVecsM,1) - min(find(propEnergyM > energyThM));
eigenVecsM = uM(:, 1:percentMarkM);
eigenValsM = sM(:, 1:percentMarkM);

energyF = zeros(size(uF,2),1);
energyThF = 0.999;
for i = 1:size(uF,2)
energyF(i) = sum(eigValsF(1:i));
end
propEnergyF = energyF./energyF(end);
percentMarkF = size(eigVecsF,1) - min(find(propEnergyF > energyThF));
eigenVecsF = uF(:, 1:percentMarkF);
eigenValsF = sF(:, 1:percentMarkF);

for i = 1:size(eigenVecsM,2)
eigenVecsM(:,i) = eigenVecsM(:,i)./sqrt(sum(eigenVecsM(:,i).^2));
end

for i = 1:size(eigenVecsF,2)
eigenVecsF(:,i) = eigenVecsF(:,i)./sqrt(sum(eigenVecsF(:,i).^2));
end

% Projection of eigenvectors on men faces from testing set.
WeightsMonM = pinv(eigenVecsM)*testMen;
WeightsFonM = pinv(eigenVecsF)*testMen;

WeightsMonF = pinv(eigenVecsM)*testWomen;
WeightsFonF = pinv(eigenVecsF)*testWomen;


% For faces other than in training set, a weight difference is calculated
% by subtracting the euclidian norms of projections. Prediction will depend
% on difference of each norm., the face is predicted to be neutral. Else, it is
% predicted to be smiling.
errorOnM = 0;
errorFaces = zeros(testFaces);
for i = 1:testFaces
    Wdif = sqrt(sum(WeightsFonM(:,i).^2)) - sqrt(sum(WeightsMonM(:,i).^2));
    if Wdif > 0
        errorOnM = errorOnM + 1;
        errorFaces(i) = 1;
    end
end
fprintf('Number of error while testing on men faces: %d out of %d\n', errorOnM, testFaces);
fprintf('Success rate: %1.3f\n\n\n', 100*(1 - errorOnM/testFaces));
% if errorOnM ~= 0
%     disp('Faces with errors:')
%     disp(find(errorFaces))
% end

errorOnF = 0;
errorFaces = zeros(testFaces);
for i = 1:testFaces
    Wdif = sqrt(sum(WeightsMonF(:,i).^2)) - sqrt(sum(WeightsFonF(:,i).^2));
    if Wdif > 0
        errorOnF = errorOnF + 1;
        errorFaces(i) = 1;
    end
end
fprintf('Number of error while testing on women faces: %d out of %d\n', errorOnF, testFaces);
fprintf('Success rate: %1.3f\n\n\n', 100*(1 - errorOnF/testFaces));
% if errorOnF ~= 0
%     disp('Faces with errors:')
%     disp(find(errorFaces))
% end

function meanMat = getMean(Mat)
mean_1 = mean(Mat, 2);
mean_2 = repmat(mean_1, 1, size(Mat,2));
meanMat = Mat - mean_2;
end