function [G, lambda, r, N] = two_triangles(g)
  % Two disjoint triangles as one network (6 nodes, 6 branches): the Laplacian, its eigenvalues, its rank and a
  % basis of its null space (Chapter 2, Exercise 3). g is the vector of the six conductances, 1 S each by default.
  if nargin < 1
    g = ones(1, 6);
  end
  fr = [1 2 3 4 5 6];
  to = [2 3 1 5 6 4];
  G = laplacian(fr, to, g, 6);
  lambda = sort(eig(G));
  r = rank(G);
  N = null(G);
end

%!test
%! [G, lambda, r, N] = two_triangles();
%! assert(sum(abs(lambda) < 1e-9), 2)             % two zero eigenvalues: two components
%! assert(lambda(3:end), [3; 3; 3; 3], 1e-9)       % a unit triangle has eigenvalues 0, 3, 3
%! assert(r, 4)                                    % rank = n - number of components = 6 - 2
%! assert(size(N), [6 2])
%! assert(G * N, zeros(6, 2), 1e-9)
%! % each null vector is constant on each triangle: the differences inside a triangle vanish
%! assert(N(1:3, :) - N([2 3 1], :), zeros(3, 2), 1e-9)
%! assert(N(4:6, :), N([5 6 4], :), 1e-9)
%!test
%! [G, lambda, r] = two_triangles([1 2 3 4 5 6]);
%! assert(sum(abs(lambda) < 1e-9), 2)
%! assert(r, 4)
%! assert(G * ones(6, 1), zeros(6, 1), 1e-12)
