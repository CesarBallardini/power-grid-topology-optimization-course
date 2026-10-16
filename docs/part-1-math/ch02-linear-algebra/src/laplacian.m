function [G, v] = laplacian(fr, to, g, n, i)
  % Weighted Laplacian G = A' * diag(g) * A of a resistor network and, if an injection vector i is
  % given, the node voltages with node 1 grounded (Chapter 2). Indices start at 1.
  m = numel(fr);
  A = zeros(m, n);
  A(sub2ind([m n], 1:m, fr)) = 1;
  A(sub2ind([m n], 1:m, to)) = -1;
  G = A' * diag(g) * A;
  if nargin > 4
    v = [0; G(2:end, 2:end) \ i(2:end)];
  end
end

%!test
%! [G, v] = laplacian([1 2 3 3], [2 3 1 4], [1 2 1 0.5], 4, [1; 0; 0; -1]);
%! assert(G, G', 1e-12)
%! assert(sort(eig(G)), [0; 0.6106; 3.091; 5.2984], 5e-5)   % one zero eigenvalue
%! assert(abs(null(G)), 0.5 * ones(4, 1), 1e-9)              % constant vector, up to sign
%! assert(G * v, [1; 0; 0; -1], 1e-12)
