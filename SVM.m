clc;
clear;
load('Svm.mat');
[m n]=size(X);
figure
y(y==0)=-1;
plot(X(y==1,1),X(y==1,2),'b*');
hold on
plot(X(y==-1,1),X(y==-1,2),'ro');

alpha=.1;
%%
b=0;
w=zeros(1,n);
lambda=.001;
f=0;
i=0;
while true
    i=i+1;
    for c=1:m
        if y(c)*(X(c,:)*w'+b)-1<0
            b=b+alpha*(y(c));
            w=w-alpha*(lambda*w-y(c)*X(c,:));
            f=1;
        end
    end
    if f==0
            break;
    else
            f=0;
    end
end
hold on
x=[0:5];
%%

plot(x,(w(1)*x+b+1)/(-1*w(2)));
plot(x,(w(1)*x+b)/(-1*w(2)));
plot(x,(w(1)*x+b-1)/(-1*w(2)));

oo=X*w'+b;
oo(oo>=0)=1;
oo(oo<0)=-1;
Error=sum(0.5*abs(y-oo))
i

%anther solution
% clc;
% clear;
% load('ex6data1.mat');
% [m n]=size(X);
% y(y==0)=-1;
% Xtrain=X(1:ceil(m*0.8),:);
% Xtest=X(ceil(m*0.8)+1:m,:);
% 
% ytrain=y(1:ceil(m*0.8),:);
% ytest=y(ceil(m*0.8)+1:m,:);
% 
% figure
% plot(Xtrain(ytrain==1,1),Xtrain(ytrain==1,2),'b*');
% hold on
% plot(Xtrain(ytrain==-1,1),Xtrain(ytrain==-1,2),'ro');
% 
% alpha=.01;
% %%
% b=0;
% w=zeros(1,n);
% t=.0000001;
% [m n]=size(Xtrain);
% for i=1:100000
%     oldw=w;
%     for c=1:m
%         if 1-ytrain(c)*(Xtrain(c,:)*w'+b)>0
%             b=b-alpha*(-1*ytrain(c));
%             w=w-alpha*(2*t*w-(1/n)*ytrain(c)*Xtrain(c,:));
%         else 
%             b=b;
%             w=w-alpha*(2*t*w);
%         end
%     end
% end
% hold on
% x=[0:5];
% %%
% 
% plot(x,(w(1)*x+b+1)/(-1*w(2)));
% plot(x,(w(1)*x+b)/(-1*w(2)));
% plot(x,(w(1)*x+b-1)/(-1*w(2)));
% %%
% oo=Xtest*w'+b;
% oo(oo>=0)=1;
% oo(oo<0)=-1;
% plot(Xtest(oo==1,1),Xtest(oo==1,2),'b*');
% hold on
% plot(Xtest(oo==-1,1),Xtest(oo==-1,2),'ro');
% Error=sum(0.5*abs(ytest-oo))

