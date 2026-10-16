function [err, denom, lodf64, lodf32] = lodf_float32(casename)
% LODF of a MATPOWER case from the double-precision PTDF and from single(full(PTDF)) (Chapter 33, Exercise 5).
%   [err, denom] = lodf_float32('case118') returns, per outaged branch, the largest absolute difference between
%   the two LODF columns and the LODF denominator 1 - (PTDF(b, f_b) - PTDF(b, t_b)). Radial branches have a zero
%   denominator and give NaN or Inf entries, which are reported as they are. Called without output arguments it
%   prints the summary quoted in the solutions. Octave has no single-precision sparse matrices, so the LODF is
%   formed with lodf_from_ptdf below, a copy of makeLODF's formula on full matrices, checked against makeLODF
%   in the test block.
  if nargin < 1
    casename = 'case118';
  end
  mpc = ext2int(loadcase(casename));
  ptdf = makePTDF(mpc);
  lodf64 = lodf_from_ptdf(mpc.branch, full(ptdf));
  lodf32 = double(lodf_from_ptdf(mpc.branch, single(full(ptdf))));
  err = max(abs(lodf64 - lodf32), [], 1)';
  nl = size(ptdf, 1);
  idx = (1:nl)';
  denom = 1 - (ptdf(sub2ind(size(ptdf), idx, mpc.branch(:, 1))) - ptdf(sub2ind(size(ptdf), idx, mpc.branch(:, 2))));
  if nargout == 0
    meshed = abs(denom) > 1e-11;
    printf('%s: %d branches, %d radial (denominator 0)\n', casename, nl, sum(~meshed));
    printf('meshed branches: largest float32 LODF error %.3e, median %.3e\n', max(err(meshed)), median(err(meshed)));
    printf('smallest meshed denominator %.3e, largest %.3e\n', min(abs(denom(meshed))), max(abs(denom(meshed))));
    [~, worst] = max(err(meshed));
    m = find(meshed);
    printf('worst outage: branch %d, denominator %.4f, error %.3e\n', m(worst), denom(m(worst)), err(m(worst)));
    printf('float32 eps %.3e; error / (eps / denominator) between %.2f and %.2f\n', eps('single'), ...
           min(err(meshed) ./ (eps('single') ./ abs(denom(meshed)))), ...
           max(err(meshed) ./ (eps('single') ./ abs(denom(meshed)))));
    printf('radial branches in float32: denominators %s\n', mat2str(double(1 - (single(ptdf(sub2ind(size(ptdf), ...
           idx(~meshed), mpc.branch(~meshed, 1)))) - single(ptdf(sub2ind(size(ptdf), idx(~meshed), ...
           mpc.branch(~meshed, 2)))))), 3));
  end
end

function lodf = lodf_from_ptdf(branch, ptdf)
% makeLODF's formula on a full PTDF of any floating-point class: H = PTDF * Cft, LODF = H ./ (1 - diag(H)').
  [nl, nb] = size(ptdf);
  i = (1:nl)';
  cft = zeros(nb, nl, class(ptdf));
  cft(sub2ind([nb, nl], branch(:, 1), i)) = 1;
  cft(sub2ind([nb, nl], branch(:, 2), i)) = -1;
  h = ptdf * cft;
  d = diag(h);
  lodf = h ./ (ones(nl, 1) * (1 - d'));
  lodf = lodf - diag(diag(lodf)) - eye(nl, class(ptdf));
end

%!test
%! if isempty(which('loadcase')), disp('MATPOWER not on the path, test skipped'); return; end
%! mpc = ext2int(loadcase('case9'));
%! ptdf = makePTDF(mpc);
%! ref = makeLODF(mpc.branch, ptdf);
%! [~, ~, lodf64] = lodf_float32('case9');
%! assert(max(abs(lodf64(:) - ref(:))) < 1e-12);
%! d = diag(lodf64);
%! assert(all(abs(d(isfinite(d)) + 1) < 1e-12));  % one radial branch gives NaN

%!test
%! if isempty(which('loadcase')), disp('MATPOWER not on the path, test skipped'); return; end
%! [err, denom] = lodf_float32('case118');
%! assert(numel(err), 186);
%! meshed = abs(denom) > 1e-11;
%! assert(sum(~meshed), 9);
%! assert(all(err(meshed) > 0));
%! assert(max(err(meshed)) < 1e-3);
%! assert(all(abs(denom(meshed)) > 1e-3));
