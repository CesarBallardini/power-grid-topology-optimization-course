function [oe, nc, worst, F1, lim] = matpower_n1_metrics(casename, load_bus, factor)
  % DC N-1 flow matrix of a MATPOWER case from makePTDF and makeLODF, with ToOp's overload energy and
  % critical branch count (Chapter 19, Exercise 5). Every limit is 1.2 times the base-case flow of the
  % unmodified case; with load_bus > 0 the load of that bus is multiplied by factor before the flows are
  % computed, the limits staying as they were. Bridges (NaN columns of the LODF) are skipped.
  % F1(:, k) holds the flows after the outage of branch k; worst(b) is the outage with the largest
  % overload of branch b (1 when it is never overloaded).
  if nargin < 2, load_bus = 0; end
  if nargin < 3, factor = 1; end
  define_constants;
  mpc = ext2int(loadcase(casename));
  nb = size(mpc.bus, 1);
  nl = size(mpc.branch, 1);
  H = makePTDF(mpc);
  L = makeLODF(mpc, H, true);                       % mask_bridge = true: NaN columns for bridges
  Cg = sparse(mpc.gen(:, GEN_BUS), 1:size(mpc.gen, 1), 1, nb, size(mpc.gen, 1));
  P = Cg * mpc.gen(:, PG) - mpc.bus(:, PD);         % MW injections; the slack absorbs the mismatch
  lim = 1.2 * abs(H * P);                           % limits from the original base case
  if load_bus > 0
    P(load_bus) = P(load_bus) - (factor - 1) * mpc.bus(load_bus, PD);
  end
  F = H * P;
  F1 = repmat(F, 1, nl) + L .* repmat(F', nl, 1);   % F + LODF(:, k) F(k), column by column
  over = max(abs(F1) - repmat(lim, 1, nl), 0);      % NaN columns (bridges) are ignored by max
  over(over < 1e-9) = 0;                            % round-off on zero-flow branches is not an overload
  [oe_b, worst] = max(over, [], 2);
  oe = sum(oe_b);
  nc = sum(oe_b > 0);
end

%!test
%! if isempty(which('makeLODF')), disp('MATPOWER not on the path, test skipped'); return; end
%! [oe, nc, worst, F1, lim] = matpower_n1_metrics('case30');
%! assert(size(F1), [41 41]);
%! bridges = all(isnan(F1), 1);
%! assert(sum(bridges), 3);                               % case30 has three radial branches (9-11, 12-13, 25-26)
%! d = diag(F1); assert(d(~bridges), zeros(sum(~bridges), 1), 1e-9);   % an outaged branch carries nothing
%! assert(oe > 0 && nc > 0 && nc <= 41);
%! [oe2, nc2] = matpower_n1_metrics('case30', 8, 2);     % doubling the 30 MW load of bus 8
%! assert(oe2 > oe);
%! [oe3, nc3] = matpower_n1_metrics('case30', 8, 1);     % factor 1 changes nothing
%! assert(oe3, oe, 1e-9); assert(nc3, nc);
