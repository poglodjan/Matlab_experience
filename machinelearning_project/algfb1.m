function [x,ObjList] = algfb1(x,goalfun,f,g,h,cons)
% x -inital data, noisy
% goalfun just to store energy
% cons - sigma tau, stepsizes

niter=1000;
size1=size(x,1);
size2=size(x,2);

y = h.dir_op(zeros(size1,size2)); % initialization of dual
ObjList=zeros(1,niter); % store some data, i.e. energy

progressbar % open the bar of progress

for i=1:niter
    x_old=x; % storage of old variable of image
    x=x-cons.tau*(g.grad(x)+h.adj_op(y));
    x=f.prox(x,cons.tau); % udpate of primal
    y = y + cons.sigma * h.dir_op(2*x-x_old);
    y= y - cons.sigma*h.prox(y/cons.sigma,1/cons.sigma); % update of dual, conjugate function
    ObjList(i)=goalfun(x,cons.lambda);
    progressbar(i/niter); % update progressbar
end

end
