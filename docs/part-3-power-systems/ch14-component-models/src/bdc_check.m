function [maxdiff, rows, cols, diffs] = bdc_check(casename)
  % Compare the plain DC matrix A' diag(1/x) A, which ignores taps, with the first output of MATPOWER's makeBdc,
  % and return the largest difference with the list of entries that differ (Chapter 14, Exercise 3).
  define_constants;
  mpc = ext2int(loadcase(casename));
  Bbus = makeBdc(mpc);

  nb = size(mpc.bus, 1);  nl = size(mpc.branch, 1);
  Cf = sparse(1:nl, mpc.branch(:, F_BUS), 1, nl, nb);
  Ct = sparse(1:nl, mpc.branch(:, T_BUS), 1, nl, nb);
  A = Cf - Ct;
  b = mpc.branch(:, BR_STATUS) ./ mpc.branch(:, BR_X);
  Bplain = A' * spdiags(b, 0, nl, nl) * A;
  D = Bbus - Bplain;
  [rows, cols, diffs] = find(abs(D) > 1e-9 .* ones(size(D)) & D ~= 0);
  diffs = full(D(sub2ind(size(D), rows, cols)));
  maxdiff = full(max(max(abs(D))));
end

%!test
%! if isempty(which('loadcase')), disp('MATPOWER not on the path, test skipped'); return; end
%! define_constants;
%! [maxdiff, rows, cols, diffs] = bdc_check('case14');
%! mpc = ext2int(loadcase('case14'));
%! k = find(mpc.branch(:, TAP));
%! assert(numel(k), 3);                                  % three transformers in IEEE 14
%! buses = unique([mpc.branch(k, F_BUS); mpc.branch(k, T_BUS)]);
%! assert(all(ismember(rows, buses)) && all(ismember(cols, buses)));   % only their buses differ
%! expected = max(abs(1 ./ mpc.branch(k, BR_X) .* (1 ./ mpc.branch(k, TAP) - 1)));
%! assert(maxdiff, expected, 1e-9);                      % the largest difference is b/tau - b of one transformer
%! assert(bdc_check('case9') < 1e-12);                   % no taps: the two matrices are identical
