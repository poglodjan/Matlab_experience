%% Zadanie 2
x1=imread('cameraman.png');
x1bw=im2gray(x1);
x1bw=im2double(x1bw);
figure
imshow(x1bw);

siz1=size(x1bw,1);
siz2=size(x1bw,2);
H=fspecial('motion',9,15); % maska filtru z zadania

H2=zeros(siz1,siz2);
H2(1:5,1:9)=H; % Bo rozmiar 5:9
H2=circshift(H2,[-4,-2]); % przenoszenie na lewy, górny róg
y=real(ifft2(fft2(H2).*fft2(x1bw)));

%losowy noise [-0.2,0.2]
wiersze = 70:160;
kolumny = 190:270;
region = y(wiersze, kolumny);
sigma = 0.2;
noise = imnoise(region, 'gaussian', 0, sigma^2);
y(wiersze, kolumny, :) = noise;
%%

figure
imshow(y);

K= @(x) real(ifft2(fft2(H2).*fft2(x)));

H2s=zeros(siz1,siz2);
H2s(1:5,1:9)=rot90(H,2);
H2s=circshift(H2s,[-4,-2]);

Ks= @(x) real(ifft2(fft2(H2s).*fft2(x))); %filtracja konwolucyjna od H2

% incjalizacja funkcja g odpowiada za gradient
g.grad= @(x) Ks(K(x)-y); 
g.beta=1; % norma l2 ma 1-Lipschitz gradient
g.fun=@(x) 1/2*sum(sum(K(x)-y).^2);
cons.lambda=0.1;
% inicjalizacja norma TV funkcja h odpowiada za norme
h.fun=@(x) fun_L2(x,cons.lambda,3); % dir 2 - obraz 2D, gradieient
h.beta=8; 

% --- forward backward --- %

% forward finite differences (with Neumann boundary conditions)
hor_forw = @(x) [x(:,2:end)-x(:,1:end-1), zeros(size(x,1),1)]; % horizontal
ver_forw = @(x) [x(2:end,:)-x(1:end-1,:); zeros(1,size(x,2))]; % vertical

h.dir_op=@(x) cat(3,hor_forw(x),ver_forw(x));

% direct and adjoint operators
hor_back = @(x) [-x(:,1), x(:,1:end-2)-x(:,2:end-1), x(:,end-1)];    % horizontal
ver_back = @(x) [-x(1,:); x(1:end-2,:)-x(2:end-1,:); x(end-1,:)];    % vertical

h.adj_op=@(x) hor_back(x(:,:,1))+ver_back(x(:,:,2));
h.prox=@(u,gamma) prox_L2(u,gamma*cons.lambda);

f.prox=@(x,tau) project_box(x,0,1);
f.fun=@(x) 0;

cons.tau=2/(g.beta+2);
cons.sigma=(1/cons.tau-g.beta/2)/h.beta;
cons.iter=500;

% ----------------------- %

[x_rec, it, time, crit]=FBPDfun(y,f,g,h,cons);

figure
imshow(x_rec)

% wykres Energii
figure; plot(time, crit);



