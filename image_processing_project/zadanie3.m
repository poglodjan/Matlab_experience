x1=imread('houseT.png');
n=200;
x1=im2double(x1);
figure
imshow(x1);

%% Maska
% vertival lines y:
rows = 15:200;
cols = 45:85;
FLambda = x1(rows, cols, :);
area_size = size(FLambda, 1) * size(FLambda, 2);
percentage = 0.65;
num_lines = floor(percentage * area_size);
positions = randperm(area_size, num_lines);
Lambda=FLambda;
Psi=FLambda;

% wartość zero na wybranych pozycjach dla każdego kanału
for kanal = 1:size(FLambda, 3)
    zmiana = FLambda(:,:,kanal);
    zmiana(positions) = 0;
    FLambda(:,:,kanal) = zmiana;
end
%Zapisanie y
y=x1;
y(rows, cols, :) = FLambda;

%Wyswietlanie y2 czerwonych
Lambda(:, 1:2:end, 1) = 255;  % czerwony kolor
Lambda(:, 1:2:end, 2) = 0;      
Lambda(:, 1:2:end, 3) = 0;      
y2=x1;
y2(rows, cols, :) = Lambda;
figure
imshow(y2);

Psi(:, 1:2:end, 1) = 0; 
Psi(:, 1:2:end, 2) = 0;      
Psi(:, 1:2:end, 3) = 0; 
%Nadpisanie y
y=x1;
y(rows, cols, :) = Psi;
%%

gradF = @(x) cat(4, x - x(:, [end, 1:(end-1)]), x - x([end, 1:(end-1)], :));
divf = @(w) w(:, [2:end, 1], 1) - w(:, :, 1) + w([2:end, 1], :, 2) - w(:, :, 2);
NormEps = @(u) sqrt(sum(u.^2, 3));
J = @(x) sum(sum(NormEps(gradF(x))));

ProxF= @(s,sigma) max(0,1-sigma./repmat(NormEps(s),[1 1 2])).*s;
ProxFs= @(s,sigma) s - sigma*ProxF(s/sigma,1/sigma);

ProxG=@(x,tau) x+ Flambda.*(y-Flambda.*x); % wzór na projekcje (prox)

sigma=10; % wsp dla kroku dualnego
tau=0.9/(8*sigma); % wsp dla kroku prymalne

niter=100;
E=zeros(1,niter);
S=zeros(1,niter); % SNR

x=y; % inicjalizcja obrazu przez zaszumiony
xbar=y; % inicjalizcja obrazu przez zaszumiony
xold=y; % inicjalizcja obrazu przez zaszumiony
s=gradF(y)*0; % macierz zerowa o wymiarach dualnych
theta=0; % 0 - bez relaksacji

for i=1:niter
    s=ProxFs( s+sigma* gradF(xbar),sigma); % krok dualny
    xold=x; % pomocnicze
    x=ProxG( x+tau*divf(s),tau );
    xbar=x+theta*(x-xold);
    E(i)=J(xbar);
    S(i)=snr(x1,xbar); % relacja snr pomiędzy orginałem i procesowanym
end

figure
imshow(xbar);

figure
plot(E); %wykres zmiany wartości funkcji celu

figure
plot(S); %wykres zmiany wartości wskaźnika SNR (stosunek sygnału do szumu)

