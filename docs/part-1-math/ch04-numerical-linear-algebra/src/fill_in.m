function [nnz_g, nnz_natural, nnz_amd, G] = fill_in(side, seed)
  % Fill-in of the sparse Cholesky factor of a grid network's reduced conductance matrix (Chapter 4, Exercise 5).
  % A side x side grid of resistors with random conductances in [0.5, 2) S, node 1 grounded. Returns nnz of
  % the reduced matrix G, of its Cholesky factor in the natural ordering and of the factor after symamd.
  % Use spy(G), spy(chol(G)) and spy(chol(G(p, p))) to see the fill-in.
  if nargin < 2
    seed = 0;
  end
  rand('seed', seed);
  n = side * side;
  idx = reshape(1:n, side, side);
  fr = [reshape(idx(1:end-1, :), [], 1); reshape(idx(:, 1:end-1), [], 1)];   % vertical and horizontal edges
  to = [reshape(idx(2:end, :), [], 1);   reshape(idx(:, 2:end), [], 1)];
  m = numel(fr);
  g = 0.5 + 1.5 * rand(m, 1);
  A = sparse([1:m, 1:m], [fr; to], [ones(m, 1); -ones(m, 1)], m, n);
  G = A(:, 2:end)' * spdiags(g, 0, m, m) * A(:, 2:end);   % reduced Laplacian, node 1 grounded
  nnz_g = nnz(G);
  nnz_natural = nnz(chol(G));
  p = symamd(G);
  nnz_amd = nnz(chol(G(p, p)));
end

%!test
%! [nnz_g, nnz_natural, nnz_amd, G] = fill_in(40);
%! assert(size(G), [1599 1599])
%! assert(nnz_g, 7835)                     % 1599 diagonal + 2 * 3118 off-diagonal entries
%! assert(issparse(G) && issymmetric(G))
%! assert(nnz_natural > nnz_g && nnz_amd > nnz_g)
%! assert(nnz_amd < nnz_natural)           % the fill-reducing ordering wins
%! R = chol(G);
%! assert(norm(R' * R - G, 'fro') / norm(G, 'fro') < 1e-12)
%!test
%! [nnz_g, nnz_natural, nnz_amd] = fill_in(10);
%! assert(nnz_g, 99 + 2 * 178)              % 180 edges, two of them at the grounded node
%! assert(nnz_amd <= nnz_natural)
