x1 = imread('parrot.bmp');
x1 = imresize(x1, [256, 256]);

x1bw = rgb2gray(x1); % Convert image to grayscale
x1bw = im2double(x1bw);

figure
imshow(x1bw);

siz1 = size(x1bw, 1);
siz2 = size(x1bw, 2);

H = fspecial('average', 3); % Filter mask

H2 = zeros(3, 3, 3); % Adjusted size for color channels
H2(1:3, 1:3, :) = repmat(H, [1, 1, 3]);
H2 = circshift(H2, [-1, -1, 0]); % Convolutional filter H2

y = real(ifft2(fft2(H2) .* fft2(x1bw)));

figure
imshow(y);

K = @(x) real(ifft2(fft2(H2) .* fft2(x)));

H2s = zeros(3, 3, 3); % Adjusted size for color channels
H2s(1:3, 1:3, :) = repmat(rot90(H, 2), [1, 1, 3]);
H2s = circshift(H2s, [-1, -1, 0]);

Ks = @(x) real(ifft2(fft2(H2s) .* fft2(x))); % Convolutional filtering with H2

% Initialize g function for gradient
g.grad = @(x) Ks(K(x) - y);
g.beta = 1; % L2 norm has 1-Lipschitz gradient
g.fun = @(x) 1/2 * sum(sum(sum((K(x) - y).^2)));

cons.lambda = 0.005;

% Initialize h function for total variation (TV) norm
h.fun = @(x) fun_L2(x, cons.lambda, 3); % dir 2 - 2D image, gradient
h.beta = 8; % 2^3 = 8

% --- Forward Backward --- %

% Forward finite differences (with Neumann boundary conditions)
hor_forw = @(x) [x(:, 2:end, :) - x(:, 1:end-1, :), zeros(size(x, 1), 1, 3)]; % Horizontal
ver_forw = @(x) [x(2:end, :, :) - x(1:end-1, :, :); zeros(1, size(x, 2), 3)]; % Vertical

h.dir_op = @(x) cat(4, hor_forw(x), ver_forw(x));

% Direct and adjoint operators
hor_back = @(x) [-x(:, 1, :), x(:, 1:end-2, :) - x(:, 2:end-1, :), x(:, end-1, :)];    % Horizontal
ver_back = @(x) [-x(1, :, :); x(1:end-2, :, :) - x(2:end-1, :, :); x(end-1, :, :)];    % Vertical

h.adj_op = @(x) hor_back(x(:, :, 1, :)) + ver_back(x(:, :, 2, :));

h.prox = @(u, gamma) prox_L2(u, gamma * cons.lambda);

cons.tau = 2 / (g.beta + 2);
cons.sigma = (1 / cons.tau - g.beta / 2) / h.beta;
cons.iter = 1000;

% ----------------------- %

[x_rec, objective] = FBPDfun(y, g, h, cons);

figure
imshow(x_rec)
