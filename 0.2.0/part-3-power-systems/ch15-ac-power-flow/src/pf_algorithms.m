function t = pf_algorithms(casename, algs)
  % Iterations, wall time and success of runpf with each power flow algorithm on one case
  % (Chapter 15, Exercise 4). Returns a struct array with fields alg, iterations, time, success.
  if nargin < 2, algs = {'NR', 'FDXB', 'GS'}; end
  mpc = loadcase(casename);
  t = struct('alg', {}, 'iterations', {}, 'time', {}, 'success', {});
  for k = 1:numel(algs)
    mpopt = mpoption('pf.alg', algs{k}, 'pf.tol', 1e-8, 'pf.gs.max_it', 5000, 'verbose', 0, 'out.all', 0);
    r = runpf(mpc, mpopt);
    t(k) = struct('alg', algs{k}, 'iterations', r.iterations, 'time', r.et, 'success', r.success);
  end
end

%!test
%! if isempty(which('runpf')), disp('MATPOWER not on the path, test skipped'); return; end
%! t = pf_algorithms('case14');
%! assert(all([t.success] == 1))
%! assert(t(1).iterations < t(2).iterations && t(2).iterations < t(3).iterations)
