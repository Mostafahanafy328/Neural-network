clc;
clear;
load('PCA.mat');
k=1;
[m n]=size(X);
plot(X(:,1),X(:,2),'*');

sigma = (1/m)*(X'*X);
[U,S,V] = svd(sigma);%singular value decomposition : sigma=U*S*V'
Ureduce=U(:,1:k);

for i=1:m
   Z(i,:)=Ureduce'*X(i,:)';
end

for i = 1:m
  nX(i, :) = (Ureduce*Z(i, :)')';
end

hold on
plot(nX(:,1),nX(:,2),'r');
