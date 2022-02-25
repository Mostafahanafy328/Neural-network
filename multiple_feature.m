clc;
clear;
data=load('ex1data2.txt');
[m n]=size(data);
y=data(:,n);
x=data(:,1:n-1);
x=[ones(m,1),x];
theta=zeros(n,1);
alpha=0.000000001;
J1=[];
while true
h=x*theta;
dx=((alpha/m)*(x'*(h-y)));
thetaa=theta-dx;
J1=[J1 (1/(2*m))*sum((h-y).^2)];
if abs(theta-thetaa)<0.0001
    break;
end
theta=thetaa;
end
figure('Name','error square method (J)');
plot(J1);
h=x*theta;
J=(1/(2*m))*sum((h-y).^2);
fprintf('Error is %.0f \n',J);
nx=[1380 3];
p=[1 , nx]*theta;
fprintf('prediction of [ %.0f , %.0f ] is %.2f \n',nx(1),nx(2),p);

extheta=inv(x'*x)*x'*y;
exh=x*extheta;
J=(1/(2*m))*sum((exh-y).^2);
exp=[1 , nx]*extheta;
fprintf('Error is %.0f \n',J);
fprintf('prediction of [ %.0f , %.0f ] is %.2f \n',nx(1),nx(2),exp);
