x1 = imread('parrot.bmp');
n = 256;
x1 = imresize(x1, [n, n]); % Resize image
%x1 = rgb2gray(x1);
x1 = im2double(x1);
figure
imshow(x1);

rho = 0.5;
Lambda = rand(n, n) > 0.5;
Psi = @(x) Lambda .* x;

y = Psi(x1); % Black squares in random positions
figure
imshow(y);

gradF = @(x) cat(3, x - x(:, [end, 1:(end-1)]), x - x([end, 1:(end-1)], :));
divf = @(w) (w(:, [2:end, 1], 1) - w(:, :, 1) + w([2:end, 1], :, 2) - w(:, :, 2)); % Adjoint of the gradient operator
NormEps = @(u) sqrt(sum(u.^2, 3)); % L2 norm for matrix elements
J = @(x) sum(sum(NormEps(gradF(x)))); % l1 norm composed

ProxF = @(s, sigma) max(0, 1 - sigma ./ repmat(NormEps(s), [1, 1, 2])) .* s;
ProxFs = @(s, sigma) s - sigma * ProxF(s / sigma, 1 / sigma);

ProxG = @(x, tau) x + Lambda .* (y - Lambda .* x); % Projection (prox) operator

sigma = 10; % Dual stepsize
tau = 0.9 / 80; % Primal stepsize

niter = 100;
E = zeros(1, niter);
S = zeros(1, niter); % SNR

x = y; % Initialize image as the noisy input
xbar = y; % Initialize image as the noisy input
xold = y; % Initialize image as the noisy input
s = gradF(y) * 0; % Zero matrix with dual dimensions
theta = 0; % Relaxation parameter (0 - no relaxation)

for i = 1:niter
    s = ProxFs(s + sigma * gradF(xbar), sigma); % Dual step
    xold = x; % Temporary variable
    x = ProxG(x + tau * divf(s), tau);
    xbar = x + theta * (x - xold);
    E(i) = J(xbar);
    S(i) = snr(x1, xbar); % SNR between the original and processed image
end

figure
imshow(xbar);
