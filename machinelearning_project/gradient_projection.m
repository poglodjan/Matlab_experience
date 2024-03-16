x1=imread('parott.bmp');
x1bw=rgb2gray(x1);
x1bw=im2double(x1bw);
a=4; % połowa szerokości
Lambda=ones(size(x1bw));
Lambda((floor(size(x1bw,1)/2)-a):(floor(size(x1bw,1)/2)+a), ...
    (floor(size(x1bw,2)/2)-a):(floor(size(x1bw,2)/2)+a))=0;
y=x1bw.*Lambda;
figure
imshow(y);

ProjD=@(x) x+ Lambda.*(y-Lambda.*x); % wzór na projekcje - uwaga produkt Kroneckera

gradF=@(x) cat(3,x-x(:,[end,1:(end-1)]),x-x([end,1:(end-1)],:));
divf=@(w) ( w(:,[2:end,1],1)-w(:,:,1)+w([2:end,1],:,2)-w(:,:,2)); % sprzężony do gradientu obrazu

epsilon=0.001; % wygładzenie normy l2 dla gradientu obrazu
NormEps=@(u) sqrt(epsilon^2+sum(u.^2,3)); % norma l2 wygładzona
J=@(x) sum(sum(NormEps(gradF(x)))); % gładka norma skomponowana z gradientem
Normalization=@(u) u./repmat(NormEps(u),[1 1 2]); % gradientr normy
gradJ=@(x) -divf(Normalization(gradF(x))); % wzór na gradient z kompozycji

tau=1.8/(1+8/epsilon)*10;
niter=2600;
E=zeros(1,niter); % magazyn dla wartości funkcji celu
x=y; % incjalizacja przez zaszumiony

for i=1:niter
    E(i)=J(x); % update wartości funkcji celu
    x=ProjD(x-tau*gradJ(x)); % update w alg gradientowym
end

figure
imshow(x);

figure
plot(E);
