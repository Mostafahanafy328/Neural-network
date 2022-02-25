%%
clc;
clear;
data=load('ex1data1.txt');
[m n]=size(data);
y=data(:,n);
x=data(:,1:n-1);
plot(x,y,'r*');
x=[ones(m,1),x];
%%
theta=rand(n,1);
alpha=0.01;
xp=4:.01:23;
J1=[];
while true
    h=x*theta;
    dx=((alpha/m)*(x'*(h-y)));
    thetaa=theta-dx;
    J1=[J1 (1/(2*m))*sum((h-y).^2)];
    if abs(theta-thetaa)<0.000000001
        break;
    end
    theta=thetaa;
end
    hold on
    h=theta(1)+theta(2)*xp;
    plot(xp,h,'b');
%%
figure('Name','error square method (J)');
plot(J1);
%%
%hold on
extheta=inv(x'*x)*x'*y
theta
%exh=extheta(1)+extheta(2)*xp;
%plot(xp,exh,'y');

nx=15;
exp=[1 , nx]*extheta;
%hold on
%plot(nx,exp,'black*');
p=[1 , nx]*theta;
%hold on
%plot(nx,p,'black*');

h=x*theta;
exh=x*extheta;
J1=(1/(2*m))*sum((h-y).^2);
fprintf('Error in gradient descent is %f \n',J1);
J=(1/(2*m))*sum((exh-y).^2);
fprintf('Error in Normal equation  is %f \n',J);
fprintf('prediction of %f in gradient descent is %f \n',nx,p);
fprintf('prediction of %f in Normal equation is %f \n',nx,exp);