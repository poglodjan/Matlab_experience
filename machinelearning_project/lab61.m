x_bar1=imread('coffee.jpg');
%x_bar1=rgb2gray(x_bar1);
x_bar=im2double(x_bar1);

figure
imshow(x_bar);

siz1=size(x_bar,1);
siz2=size(x_bar,2);

H = fspecial('prewitt');

H2=zeros(siz1,siz2);
H2(1:3,1:3)=H;
H2=circshift(H2,[-1,-1]);

Gk= fft2(H2); % fast fourier transform 2d for blur
K = @(x) real(ifft2(Gk.*fft2(x))); % blurring for image proceded


A_dir= @(x) K(x); % noising image

Hs=rot90(H,2); % conjugate kernel but not the proper size

H2s=zeros(siz1,siz2);
H2s(1:3,1:3)=Hs;
H2s=circshift(H2s,[-1,-1]);

Gks= fft2(H2s); % fast fourier transform 2d for blur adjectment
Ks = @(x) real(ifft2(Gks.*fft2(x))); % blurring for image proceded

A_adj= @(x) Ks(x); % adjoint


y=A_dir(x_bar);

figure
imageplot(clamp(y,0,1))

g.grad = @(x) A_adj(A_dir(x)-y);
g.beta=1;
g.fun= @(x) 1/2 * sum(sum(A_dir(x)-y).^2); % data fidelity function

% TV function
h.fun= @(x,lam) fun_L2(x,lam,3); 
h.beta=8;

% forward finite differences (with periodical boundary conditions)
hor_forw = @(x) [x(:,2:end)-x(:,1:end-1), x(:,1)-x(:,end) ]; % horizontal
ver_forw = @(x) [x(2:end,:)-x(1:end-1,:); x(1,:)-x(end,:) ]; % vertical

% backward finite differences (with peridical boundary conditions)
hor_back = @(x) [x(:,end)-x(:,1), x(:,1:end-2)-x(:,2:end-1), x(:,end-1)-x(:,end)];    % horizontal
ver_back = @(x) [x(end,:)-x(1,:); x(1:end-2,:)-x(2:end-1,:); x(end-1,:)-x(end,:)];    % vertical

h.dir_op=@(x) cat(3,hor_forw(x),ver_forw(x)); % result in dual space
h.adj_op=@(u) hor_back(u(:,:,1))+ver_back(u(:,:,2));

cons.lambda=0.005;
h.prox= @(u,gamma) prox_L2(u,gamma*cons.lambda);

f.prox = @(x,tau) project_box(x,0,1);

goalfun=@(x,lam) g.fun(x)+lam*h.fun(x,lam);

cons.tau=2/(g.beta+2);
cons.sigma=(1/cons.tau-g.beta/2)/h.beta;

[x_rec,Objective]=algfb1(y,goalfun,f,g,h,cons);

figure
imshow(x_rec)

