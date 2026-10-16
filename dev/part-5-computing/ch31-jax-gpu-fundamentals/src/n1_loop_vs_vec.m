function [F_loop, F_vec, t_loop, t_vec] = n1_loop_vs_vec(casename)
  % N-1 flows of every single branch outage of a MATPOWER case, computed twice: a for loop over the
  % outages and one vectorized matrix expression, each timed (Chapter 31, Exercise 5).
  % Row k of F is the flow vector after the outage of branch k; bridge rows are NaN in both versions.
  define_constants;
  mpc = ext2int(loadcase(casename));
  H  = makePTDF(mpc);                                   % PTDF, slack = reference bus
  Pbus = real(makeSbus(mpc.baseMVA, mpc.bus, mpc.gen)) * mpc.baseMVA;   % net injections, MW
  f0 = H * Pbus;                                        % base-case flows
  f = mpc.branch(:, F_BUS);
  t = mpc.branch(:, T_BUS);
  nl = size(mpc.branch, 1);

  tic;
  F_loop = nan(nl, nl);
  for k = 1:nl                                          % one outage at a time, as a Python loop would
    col = H(:, f(k)) - H(:, t(k));
    denom = 1 - col(k);
    if abs(denom) < 1e-11
      continue;                                         % bridge: islands the grid, row stays NaN
    end
    lodf = col / denom;
    lodf(k) = -1;
    F_loop(k, :) = (f0 + lodf * f0(k))';
  end
  t_loop = toc;

  tic;
  col = H(:, f) - H(:, t);                              % column k: PTDF difference of outage k, all at once
  denom = 1 - diag(col)';                               % 1 - (H(k, f_k) - H(k, t_k)) as a row vector
  L = col ./ denom;                                     % broadcasting divides each column by its denominator
  L(1:nl + 1:end) = -1;                                 % the outaged branch loses its flow
  L(:, abs(denom) < 1e-11) = NaN;                       % bridges (makeLODF(mpc, H, true) fails on case118 in 8.1)
  F_vec = f0' + L' .* f0;                               % F(k, a) = f0(a) + L(a, k) f0(k), by broadcasting
  t_vec = toc;
end

%!test
%! if isempty(which('loadcase')), disp('MATPOWER not on the path, test skipped'); return; end
%! [Fl, Fv, tl, tv] = n1_loop_vs_vec('case118');
%! assert(size(Fl), [186 186]); assert(size(Fv), [186 186])
%! ok = all(isfinite(Fl), 2);
%! assert(sum(~ok) >= 1)                                % IEEE 118 has bridges
%! assert(isequal(ok, all(isfinite(Fv), 2)))            % both versions flag the same outages
%! assert(Fv(ok, :), Fl(ok, :), 1e-8)
%! assert(tl > 0 && tv > 0)
%! fprintf('case118: loop %.4f s, vectorized %.4f s, ratio %.1f\n', tl, tv, tl / tv)
