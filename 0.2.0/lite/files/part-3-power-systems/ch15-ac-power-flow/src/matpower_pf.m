function [results, results_fd] = matpower_pf(casename)
  % Solve a MATPOWER case with Newton-Raphson (enforcing reactive limits) and with the XB fast decoupled
  % method, and return both result structs (Chapter 15).
  mpc = loadcase(casename);
  mpopt = mpoption('pf.alg', 'NR', 'pf.enforce_q_lims', 1, 'verbose', 0, 'out.all', 0);
  results = runpf(mpc, mpopt);                   % results.success is 1 if converged
  mpopt = mpoption(mpopt, 'pf.alg', 'FDXB');     % fast decoupled, XB version
  results_fd = runpf(mpc, mpopt);
end

%!test
%! if isempty(which('runpf')), disp('MATPOWER not on the path, test skipped'); return; end
%! [r, rfd] = matpower_pf('case14');
%! assert(r.success == 1 && rfd.success == 1)
%! assert(r.bus(:, 8), rfd.bus(:, 8), 1e-6)       % same voltage magnitudes (column VM)
