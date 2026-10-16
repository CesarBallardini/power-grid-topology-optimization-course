function [Zref, xstar] = trafo_equivalents(Z, V1, V2, x12, x13, x23)
  % Impedance Z on the V2 side seen from the V1 side of an ideal transformer, Z * (V1 / V2)^2, and the
  % star equivalent [x1 x2 x3] of a three-winding transformer from its pairwise short-circuit reactances
  % (Chapter 11, Exercises 1 and 3).
  Zref = Z * (V1 / V2)^2;
  xstar = [x12 + x13 - x23, x12 + x23 - x13, x13 + x23 - x12] / 2;
end

%!test
%! [Zref, xstar] = trafo_equivalents(10, 380, 110, 0.10, 0.25, 0.20);
%! assert(Zref, 119.3, 0.05)
%! assert(xstar, [0.075 0.025 0.175], 1e-12)
%! assert(xstar(1) + xstar(2), 0.10, 1e-12)      % each pair of star branches gives back a pairwise reactance
%! assert(xstar(1) + xstar(3), 0.25, 1e-12)
%! assert(xstar(2) + xstar(3), 0.20, 1e-12)
