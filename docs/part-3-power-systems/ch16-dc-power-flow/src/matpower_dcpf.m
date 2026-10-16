function [pf, Bbus, Bf] = matpower_dcpf(casename)
  % DC power flow of a MATPOWER case: the Bbus and Bf matrices from makeBdc and the from-end flows (MW)
  % of rundcpf (Chapter 16).
  define_constants;
  mpc = ext2int(loadcase(casename));
  [Bbus, Bf, Pbusinj, Pfinj] = makeBdc(mpc);
  results = rundcpf(mpc, mpoption('out.all', 0, 'verbose', 0));
  pf = results.branch(:, PF);
end

%!test
%! if isempty(which('rundcpf')), disp('MATPOWER not on the path, test skipped'); return; end
%! [pf, Bbus, Bf] = matpower_dcpf('case14');
%! assert(size(Bbus), [14 14]); assert(size(Bf), [20 14])
%! assert(abs(sum(Bbus, 2)) < 1e-9)                 % every row of Bbus sums to zero
%! assert(numel(pf), 20)
