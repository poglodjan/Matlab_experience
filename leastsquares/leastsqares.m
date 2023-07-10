function [p0,p1]=leastsqares(x,y)
    n=size(x,2);
    X=ones(n,2);
    X(:,2)=x;
    Y=y';
    P=(X'*X)\X'*Y;
    p0=P(1);
    p1=P(2);
    hold on
    plot(x,y,'.');
    plot([x(1),x(end)],[p0+p1*x(1),p0+p1*x(end)]);
end



