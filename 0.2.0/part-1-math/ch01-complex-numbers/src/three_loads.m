function T = three_loads(V, Z)
  % Current and complex power of several loads Z (a vector of impedances) on the same source V (Chapter 1,
  % Lab step 4). One row per load: [R, X, real(I), imag(I), P, Q, abs(S), hypot(P, Q)]; sign(Q) is the sign of X.
  T = zeros(numel(Z), 8);
  for k = 1:numel(Z)
    [I, S] = phasor_power(V, Z(k));
    T(k, :) = [real(Z(k)), imag(Z(k)), real(I), imag(I), real(S), imag(S), abs(S), hypot(real(S), imag(S))];
  end
end

%!test
%! T = three_loads(230, [40 + 30i, 40 - 30i, 50]);
%! assert(T(:, 7), T(:, 8), 1e-9)                       % |S| = sqrt(P^2 + Q^2) for every load
%! assert(sign(T(:, 6)), [1; -1; 0])                     % Q > 0 inductive, Q < 0 capacitive, Q = 0 resistive
%! assert(T(1, 5:6), [846.4, 634.8], 1e-9)
%! assert(T(2, 5:6), [846.4, -634.8], 1e-9)
%! assert(T(3, 5:7), [1058, 0, 1058], 1e-9)
