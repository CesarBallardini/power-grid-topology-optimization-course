function [shift_deg, du] = tap_phase_shift(Vn, n, s, phi_deg)
  % Phase shift in degrees of a tap changer that adds du = Vn * n * s / 100 at the step angle phi_deg to the
  % nominal voltage Vn (Chapter 1, Exercise 4): atan2(du sin(phi), Vn + du cos(phi)), the formula ToOp uses.
  du = Vn * n * s / 100;
  phi = phi_deg * pi / 180;
  shift_deg = atan2(du * sin(phi), Vn + du * cos(phi)) * 180 / pi;
end

%!test
%! [shift, du] = tap_phase_shift(400, 5, 1, 90);
%! assert(du, 20, 1e-12)
%! assert(shift, 2.86, 5e-3)
%! assert(shift, angle(1 + du / 400 * exp(1i * pi / 2)) * 180 / pi, 1e-12)   % same as arg(1 + (du/Vn) e^{j phi})
%!test
%! assert(tap_phase_shift(400, 5, 1, 60), 2.42, 5e-3)
%!test
%! assert(tap_phase_shift(400, 0, 1, 60), 0, 1e-12)   % neutral tap: no shift
