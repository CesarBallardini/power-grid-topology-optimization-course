function [Z, I, S, pf, C] = rl_load(R, L, V, f, pf_target)
  % Series RL load on a sinusoidal supply: impedance, current, complex power, power factor and the
  % parallel capacitance that raises the power factor to pf_target (Chapter 9, Exercise 1).
  w = 2 * pi * f;
  Z = R + 1i * w * L;
  I = V / Z;
  S = V * conj(I);
  pf = real(S) / abs(S);
  Qc = imag(S) - real(S) * tan(acos(pf_target));
  C = Qc / (w * V^2);
end

%!test
%! [Z, I, S, pf, C] = rl_load(10, 50e-3, 230, 50, 0.95);
%! assert(Z, 10 + 15.708i, 1e-3)
%! assert(abs(I), 12.35, 5e-3)
%! assert([real(S), imag(S)], [1526, 2396], 1)
%! assert(pf, 0.537, 5e-4)
%! assert(C * 1e6, 114, 0.1)
%! Zc = 1 / (1 / Z + 1i * 2 * pi * 50 * C);           % load with the capacitor in parallel
%! Sc = 230 * conj(230 / Zc);
%! assert(real(Sc) / abs(Sc), 0.95, 1e-9)
