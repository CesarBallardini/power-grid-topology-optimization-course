function Minv_sm = sherman_morrison(Minv, Ar, g, k)
  % Inverse of the reduced Laplacian after removing branch k, by a rank-1 update of Minv = inv(Gr)
  % (Chapter 4). Ar is the reduced incidence matrix (branches x ungrounded nodes), g the conductances.
  a = Ar(k, :)';             % row k as a column vector
  u = Minv * a;
  den = 1 - g(k) * (a' * u); % zero exactly when branch k is a bridge
  Minv_sm = Minv + g(k) * (u * u') / den;
end

%!test
%! fr = [1 1 2 2 3 4 5]; to = [2 3 3 4 5 5 6]; g = [10 5 8 4 6 3 2]; m = numel(fr); n = 6;
%! A = zeros(m, n); A(sub2ind([m n], 1:m, fr)) = 1; A(sub2ind([m n], 1:m, to)) = -1;
%! Ar = A(:, 2:end);                       % ground node 1
%! Minv = inv(Ar' * diag(g) * Ar);
%! k = 4;                                   % branch (2, 4)
%! keep = (1:m) ~= k;
%! fresh = inv(Ar(keep, :)' * diag(g(keep)) * Ar(keep, :));
%! assert(sherman_morrison(Minv, Ar, g, k), fresh, 1e-12)
%! assert(1 - g(7) * (Ar(7, :) * Minv * Ar(7, :)'), 0, 1e-12)   % branch (5, 6) is a bridge
