% Zadanie 1, funkcja f

f=@(x) 2*(x(1)-3)^2+(x(2)-4)^4;
xx=linspace(-5,5,100);
yy=linspace(-4,4,100);
[u,v]=meshgrid(yy,xx); % Tworzymy siatke i F oblicza wartosci na siatce
F=2*(u-3).^2+(v-4).^4; % Wartość funkcji F jest obliczana dla każdego punktu (u, v)

gradF=@(x)[4*x(1)-12,4*(x(2)-4)^3]; % gradient funkcji f w punkcie x,y
x=[-3,2]; % punkt startowy
niter=400; % liczba iteracji
tau=0.001; % długość kroku
E=zeros(1,niter); % magazyn dla wartości funkcji
D=zeros(1,niter); % magazyn odległość od rozwiązania
X=zeros(2,niter);  % magazyn punktów pośrednich

for i=1:niter
    % sprawdzam warunki dla (x,y) na D
    if (x(1)<=2 && x(1) >=-2) || (x(2)<=1 && x(2) >=-1)
        X(:,i)=x;
    else
        X(:,i)=x/norm(x); %znormalizowanie
    end
    E(i)=f(x);
    D(i)=norm(x); % norma l2 Euklidesowa do 0,0
    x=x-tau*gradF(x);
end

% Przemieszczanie punktowe w kierunku minimalnej wartości funkcji F
figure
imagesc(yy,xx,F);
hold on
plot(X(1,:),X(2,:),'k-');

% Zmiana wartości f w trakcie kolejnych iteracji
figure
plot(E);

% Zmiana odległości punktu x od (0.0)
figure
plot(D);
disp(x);
