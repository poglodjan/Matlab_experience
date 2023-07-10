%Gradient descent to function
eta=10;
f=@(x)(0.5*x(1)^2+0.5*eta*x(2)^2); % funkcja
t=linspace(-1,1,100);
[u,v]=meshgrid(t,t); % Tworzymy siatke i F oblicza wartosci na siatce
F=0.5*u.^2+0.5*eta*v.^2; % Wartość funkcji F jest obliczana dla każdego punktu (u, v)
imagesc(t,t,F);

gradF=@(x)[x(1),eta*x(2)]; %gradF(x) zwraca gradient funkcji f w punkcie x
x=[0.5,0.5]; % punkt startowy
niter=200; % liczba iteracji
tau=0.5/eta;
E=zeros(1,niter); % magazyn dla wartości funkcji
D=zeros(1,niter); % magazyn odległość od rozwiązania
X=zeros(2,niter);  % magazyn punktów pośrednich
for i=1:niter
    X(:,i)=x;
    E(i)=f(x);
    D(i)=norm(x); % norma l2 Euklidesowa do 0,0
    x=x-tau*gradF(x);
end

% 1 jak punkt startowy przemieszcza się w kierunku minimalnej wartości funkcji F
figure
imagesc(t,t,F);
hold on
plot(X(1,:),X(2,:),'k-');

% 2  jak wartość funkcji zmienia się w trakcie kolejnych iteracji
figure
plot(E);

% 3 Ten wykres przedstawia zmianę odległości punktu x od (0.0)
figure
plot(D);

% 4 zaznacza odnalezione minimum:
plot(x(1), x(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);