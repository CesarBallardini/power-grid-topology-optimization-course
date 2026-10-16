function [H, L, err] = matpower_lodf(casename)
  % PTDF and LODF of a MATPOWER case, and the largest difference between the LODF computed from a
  % single-slack PTDF and from a distributed-slack PTDF, which must be round-off (Chapter 17).
  mpc = ext2int(loadcase(casename));
  H  = makePTDF(mpc);              % single slack: the reference bus
  L  = makeLODF(mpc, H, true);     % mask_bridge = true: NaN columns for bridges
  nb = size(mpc.bus, 1);
  w  = ones(nb, 1) / nb;           % a distributed slack
  Hw = makePTDF(mpc, w);
  err = max(max(abs(makeLODF(mpc, Hw) - makeLODF(mpc, H))));   % LODF is slack independent
end

%!test
%! if isempty(which('makeLODF')), disp('MATPOWER not on the path, test skipped'); return; end
%! [H, L, err] = matpower_lodf('case14');
%! assert(size(H), [20 14]); assert(size(L), [20 20])
%! assert(err < 1e-9)
%! assert(diag(L)(~isnan(diag(L))), -ones(sum(~isnan(diag(L))), 1), 1e-9)   % outaged branch loses its flow
