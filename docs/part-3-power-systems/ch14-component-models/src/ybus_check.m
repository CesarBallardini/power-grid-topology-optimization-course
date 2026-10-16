function err = ybus_check(casename)
  % Rebuild the bus admittance matrix of a MATPOWER case from the branch model, Ybus = Cf' (Yff Cf + Yft Ct)
  % + Ct' (Ytf Cf + Ytt Ct) + Ysh, and return the largest difference against makeYbus (Chapter 14).
  define_constants;
  mpc = ext2int(loadcase(casename));
  [Ybus, Yf, Yt] = makeYbus(mpc);

  nb = size(mpc.bus, 1);  nl = size(mpc.branch, 1);
  Cf = sparse(1:nl, mpc.branch(:, F_BUS), 1, nl, nb);
  Ct = sparse(1:nl, mpc.branch(:, T_BUS), 1, nl, nb);
  stat = mpc.branch(:, BR_STATUS);
  ys = stat ./ (mpc.branch(:, BR_R) + 1j * mpc.branch(:, BR_X));
  bc = stat .* mpc.branch(:, BR_B);
  tap = ones(nl, 1);  k = find(mpc.branch(:, TAP));  tap(k) = mpc.branch(k, TAP);
  tap = tap .* exp(1j * pi / 180 * mpc.branch(:, SHIFT));
  Ytt = ys + 1j * bc / 2;  Yff = Ytt ./ (tap .* conj(tap));
  Yft = -ys ./ conj(tap);  Ytf = -ys ./ tap;
  Ysh = (mpc.bus(:, GS) + 1j * mpc.bus(:, BS)) / mpc.baseMVA;
  D = @(v) spdiags(v, 0, numel(v), numel(v));
  Y2 = Cf.' * (D(Yff) * Cf + D(Yft) * Ct) + Ct.' * (D(Ytf) * Cf + D(Ytt) * Ct) + D(Ysh);
  err = full(max(max(abs(Ybus - Y2))));   % expect round-off only
end

%!test
%! if isempty(which('makeYbus')), disp('MATPOWER not on the path, test skipped'); return; end
%! assert(ybus_check('case14') < 1e-12)
%! assert(ybus_check('case57') < 1e-12)
