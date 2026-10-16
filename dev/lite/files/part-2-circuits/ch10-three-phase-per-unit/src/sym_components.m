function [V012, A] = sym_components(Vabc)
  % Symmetrical components V012 = A \ Vabc of a three-phase set, with a = exp(2i pi / 3) and
  % A = [1 1 1; 1 a^2 a; 1 a a^2] (Chapter 10, Exercise 5). Vabc is a column of three phasors.
  a = exp(2i * pi / 3);
  A = [1 1 1; 1 a^2 a; 1 a a^2];
  V012 = A \ Vabc(:);
end

%!test
%! a = exp(2i * pi / 3);
%! V012 = sym_components([1; a^2; a]);                 % balanced set: positive sequence only
%! assert(abs(V012), [0; 1; 0], 1e-12)
%! V012 = sym_components([1; 0.9 * a^2; a]);           % phase b at 0.9 pu
%! assert(abs(V012), [0.0333; 0.9667; 0.0333], 5e-5)
%! [V012, A] = sym_components([1; 0.9 * a^2; a]);
%! assert(A * V012, [1; 0.9 * a^2; a], 1e-12)          % the transform inverts
