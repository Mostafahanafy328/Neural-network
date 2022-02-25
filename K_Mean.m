clc;
clear;
load('MEAN.mat');
[m n]=size(X);
k=input('Enter number of centroid = ');
for I=1:10
q=randperm(m);
c=X(q(1:k),:);
while true
    for i=1:m
        for j=1:k
            d(i,j)=norm(X(i,:)-c(j,:));
        end
    end
    [v p]=min(d(:,1:k)');
    if size(d,2)==k+1
        length(find(d(:,k+1)~=p'))
        if length(find(d(:,k+1)~=p'))==0
            break;
        end
    end
    d(:,k+1)=p';
    for i=1:k
        c(i,:)=mean(X(find(d(:,k+1)==i),:));
    end

end
    figure
    for i=1:k
        plot(X(find(d(:,k+1)==i),1),X(find(d(:,k+1)==i),2),'*');
        hold on
    end

end

