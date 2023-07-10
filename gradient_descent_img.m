%Gradient descent to img
x3=imread('coffee.jpg');
x3bw=rgb2gray(x3);
x3bw=imresize(x3bw,[256,256]);
imshow(x3bw);
x3bw=im2double(x3bw);
%imshow(x3bw);
gradF=@(x) cat(3,x-x(:,[end,1:(end-1)]),x-x([end,1:(end-1)],:));
v=gradF(x3bw);
%figure
%imshow(abs(v(:,:,1)));
%figure
%imshow(abs(v(:,:,2)));

% dywergencja na obraz (z gradientu):
divf=@(w) ( w(:,[2:end,1],1)-w(:,:,1)+w([2:end,1],:,2)-w(:,:,2)); % sprzężony do gradientu obrazu
figure
imshow(abs(divf(v)));  
sigma=0.1;

y=x3bw+0.1*rand(256); % dodanie losowego szumu o wartości max sigma
figure
imshow(y);
lambda=0.3/5; % współczynnik do istnotności gradientu
epsilon=0.001; % wygładzenie normy l2 dla gradientu obrazu
NormEps=@(u) sqrt(epsilon^2+sum(u.^2,3)); % norma l2 wygładzona
J=@(x) sum(sum(NormEps(gradF(x)))); % gładka norma skomponowana z gradientem
f=@(x) 1/2*norm(x-y)^2+lambda*J(x); % cała funkcja celu
Normalization=@(u) u./repmat(NormEps(u),[1 1 2]); % gradientr normy
gradJ=@(x) -divf(Normalization(gradF(x))); % wzór na gradient z kompozycji
gradfun=@(x) (x-y)+ lambda*gradJ(x); % gradient funkcji celu

tau=1.8/(1+lambda*8/epsilon);
niter=500;
E=zeros(1,niter); % magazyn dla wartości funkcji celu
x=y; % incjalizacja przez zaszumiony

for i=1:niter
    E(i)=f(x); % update wartości funkcji celu
    x=x-tau*gradfun(x); % update w alg gradientowym
end

figure
imshow(x);

figure
plot(E);