function [I, S] = phasor_power(V, Z)
  % One load on a source: phasor current I = V / Z and complex power S = V * conj(I) (Chapter 1).
  I = V / Z;
  S = V * conj(I);
end

%!test
%! [I, S] = phasor_power(230, 40 + 30i);
%! assert([abs(I), angle(I) * 180 / pi], [4.6, -36.87], 0.005)
%! assert(S, 846.4 + 634.8i, 1e-9)
%! assert([abs(S), hypot(real(S), imag(S))], [1058, 1058], 1e-9)
