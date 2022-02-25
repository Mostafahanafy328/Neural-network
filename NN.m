clc;
clear;
load('NN.mat');
[m n]=size(X);
l=4;%input('enter number of layer = ');
nn=[72 52 20 17];%input('enter number of node in each layer : ');
Y=y;
y=zeros(m,10);
for i=1:m
    y(i,Y(i))=1;
end
if(length(nn)==l)

theta=zeros(max(max(nn),size(y,2)),max(max(nn),n)+1,l+1);
delta=zeros(max(max(nn),size(y,2)),max(max(nn),n)+1,l+1);

theta(1:nn(1),1:size(X,2)+1,1)=rand(nn(1),size(X,2)+1);
for i=2:l
    theta(1:nn(i),1:nn(i-1)+1,i)=rand(nn(i),nn(i-1)+1);
end

theta(1:size(y,2),1:nn(length(nn))+1,l+1)=rand(size(y,2),nn(length(nn))+1);
alpha=.1;
a=[ones(m,1) X];
for j=1:50
for i=1:m
    a1=(a(i,:))';
    b=zeros(max(max(nn),size(X,2))+1,l+2);
    b(1:size(a1),1)=a1;
    for k=1:l+1
        if k==1
            z=theta(1:nn(k),1:n+1,k)*a1;
        elseif k==l+1
            z=theta(1:size(y,2),1:nn(l)+1,k)*a2;
        else            
            z=theta(1:nn(k),1:nn(k-1)+1,k)*a2;
        end
            a2=1./(1+exp(-z));
            a2=[1; a2];
            b(1:size(a2),k+1)=a2;
    end
    h(i,:)=a2(2:length(a2));

c=length(nn);
E=zeros(max(max(nn),size(X,2))+1,l+1);
    for k=1:l+1 
        if k==1
            E1=a2(2:length(a2),1)-y(i,:)';
        else
            E2=(theta(1:t,1:nn(l-k+2)+1,l-k+3)'*E(1:t,l-k+3)).*(b(1:nn(c)+1,l-k+3)).*(1-b(1:nn(c)+1,l-k+3));
            E1=E2(2:length(E2),1);
            c=c-1;
        end
        t=length(E1);
        E(1:t,l-k+2)=E1;
    end
    
    for k=1:l+1  
        if k==1
            delta(1:nn(k),1:size(X,2)+1,k)=delta(1:nn(k),1:size(X,2)+1,k)+E(1:nn(k),k)*b(1:size(X,2)+1,k)';
        elseif k==l+1
            delta(1:size(y,2),1:nn(k-1)+1,k)=delta(1:size(y,2),1:nn(k-1)+1,k)+E(1:size(y,2),k)*b(1:nn(k-1)+1,k)';
        else            
            delta(1:nn(k),1:nn(k-1)+1,k)=delta(1:nn(k),1:nn(k-1)+1,k)+E(1:nn(k),k)*b(1:nn(k-1)+1,k)';
        end
    end
end
J(j)=(1/m)*sum(sum(((-y.*log(h))-((1-y).*log(1-h)))));

theta=theta-(alpha/m)*delta;

delta=zeros(max(max(nn),size(y,2)),max(max(nn),n)+1,l+1);

end
figure;
plot(J);
else
    fprintf('Error \n');
end
    