clc;
clear;
data1=load('ex1data1.txt');
[m,n]=size(data1);
data=[ones(m,1),data1(:,1),data1(:,1).^2,data1(:,1).^3,data1(:,1).^4,data1(:,2)];
[m,n]=size(data);
training=data(1:ceil(m*0.6),:);
cv=data(ceil(m*0.6)+1:ceil(m*0.8),:);
test=data(ceil(m*0.8)+1:m,:);


y=training(:,n);
x=training(:,1:n-1);
[m,n]=size(x);
 theta=zeros(n,n-1);
alpha=0.0000000001;
for i=1:n-1
    X=x(:,1:i+1);
while true
h=X*theta(1:i+1,i);
dx=((alpha/m)*(X'*(h-y)));
thetaa=theta(1:i+1,i)-dx;
if abs(theta(1:i+1,i)-thetaa)<0.00000001
    break;
end
theta(1:i+1,i)=thetaa;
end
theta(1:i+1,i)=thetaa;
end
%%
h1=x*theta;
J9=(1/(2*m))*sum((h1-y).^2);
%%
[m,n]=size(cv);
ycv=cv(:,n);
xcv=cv(:,1:n-1);

h1=xcv*theta;
Jcv=(1/(2*m))*sum((h1-ycv).^2);
[v index]=min(Jcv);
%%
J99=(1/(2*m))*sum((h1-ycv).^2);
figure;
plot(J9,'r');
hold on
plot(J99,'b');
%%
[m,n]=size(test);
ytest=test(:,n);
xtest=test(:,1:n-1);

h=xtest*theta(:,index);
J=(1/(2*m))*sum((h-ytest).^2);
fprintf('Error is %f n',J);


xp=4:01:23;
x=[ones(1,length(xp));xp;xp.^2;xp.^3;xp.^4];
h=theta(:,index)'*x;

[m,n]=size(data1);
y=data1(:,n);
x=data1(:,1:n-1);
figure;
plot(x,y,'r*');
hold on
plot(xp,h);
