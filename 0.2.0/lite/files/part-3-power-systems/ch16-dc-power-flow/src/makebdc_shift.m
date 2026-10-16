function [err_bbus, pfinj0, pfinj, err_shift] = makebdc_shift(casename, branch_row, shift_deg)
  % Check makeBdc against A' diag(b) A with b = 1/(x tau), and Pfinj against -b * shift (rad), after adding a
  % phase shift to one branch (Chapter 16, Exercise 4). Returns the largest entry differences and both Pfinj.
  define_constants;
  mpc = ext2int(loadcase(casename));
  [Bbus, ~, ~, pfinj0] = makeBdc(mpc);
  nb = size(mpc.bus, 1); nl = size(mpc.branch, 1);
  tau = mpc.branch(:, TAP); tau(tau == 0) = 1;
  b = 1 ./ (mpc.branch(:, BR_X) .* tau);
  A = sparse([1:nl 1:nl], [mpc.branch(:, F_BUS); mpc.branch(:, T_BUS)], [ones(nl, 1); -ones(nl, 1)], nl, nb);
  err_bbus = full(max(max(abs(Bbus - A' * diag(b) * A))));
  mpc.branch(branch_row, SHIFT) = shift_deg;
  [~, ~, Pbusinj, pfinj] = makeBdc(mpc);
  expected = zeros(nl, 1); expected(branch_row) = -b(branch_row) * shift_deg * pi / 180;
  err_shift = max(abs(pfinj - expected));
  err_shift = max(err_shift, full(max(abs(Pbusinj - A' * expected))));   % the bus injections are A' Pfinj
end

%!test
%! if isempty(which('makeBdc')), disp('MATPOWER not on the path, test skipped'); return; end
%! [err_bbus, pfinj0, pfinj, err_shift] = makebdc_shift('case14', 8, 3);
%! assert(err_bbus < 1e-9)
%! assert(all(pfinj0 == 0))
%! assert(err_shift < 1e-12)
%! assert(sum(pfinj ~= 0), 1)
