function wf = train_BiPCF(params, xf, yf, s, xf_p, wf_p)

Sxy = cellfun(@(xf, yf) bsxfun(@times, xf, conj(yf)), xf, yf, 'UniformOutput', false);
Sxx = cellfun(@(xf) bsxfun(@times, xf, conj(xf)), xf, 'UniformOutput', false);
x_fuse = cellfun(@(xf, xf_p) xf + xf_p, xf, xf_p, 'UniformOutput', false);
Sxx_fuse = cellfun(@(x_fuse) bsxfun(@times, x_fuse, conj(x_fuse)), x_fuse, 'UniformOutput', false);

% feature size
sz = cellfun(@(xf) size(xf), xf, 'UniformOutput', false);
N = cellfun(@(sz) sz(1) * sz(2), sz, 'UniformOutput', false);

% initialize hs
hf = cellfun(@(sz) zeros(sz), sz, 'UniformOutput', false);

% initialize lagrangian multiplier
zetaf = cellfun(@(sz) zeros(sz), sz, 'UniformOutput', false);

% penalty
mu = params.mu;
beta = params.beta;
mu_max = params.mu_max;
lambda = params.lambda;
gamma = params.gamma;

iter = 1;
while (iter <= params.admm_iterations)

    wf = cellfun(@(Sxy, wf_p, Sxx, Sxx_fuse, hf, zetaf) ...
        bsxfun(@rdivide, ...
		Sxy + bsxfun(@times, wf_p, gamma * Sxx_fuse) + mu * hf - zetaf, ...
		Sxx + gamma * Sxx_fuse + mu), ...
		Sxy, wf_p, Sxx, Sxx_fuse, hf, zetaf, 'UniformOutput', false);
		
    hf = cellfun(@(wf, zetaf) fft2(bsxfun(@rdivide, real(ifft2(mu * wf + zetaf)), lambda * s.^2 + mu)), wf, zetaf, 'UniformOutput', false);
    
    zetaf = cellfun(@(zetaf, wf, hf) zetaf + mu * (wf - hf), zetaf, wf, hf, 'UniformOutput', false);
    
    mu = min(mu_max, beta * mu);
    
    iter = iter + 1;
end

