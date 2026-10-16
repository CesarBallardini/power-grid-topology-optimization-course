function [H, Hw, dcol, dflow] = ptdf_slack_case30(casename)
  % PTDF of a MATPOWER case for a single slack (the reference bus) and for a slack distributed in proportion
  % to the generator outputs; the largest change per PTDF column and the largest change of the base-case
  % flows (Chapter 17, Exercise 4).
  define_constants;
  mpc = ext2int(loadcase(casename));
  nb = size(mpc.bus, 1);
  H = makePTDF(mpc);                                 % single slack
  w = zeros(nb, 1);
  on = mpc.gen(:, GEN_STATUS) > 0;
  w(mpc.gen(on, GEN_BUS)) = w(mpc.gen(on, GEN_BUS)) + mpc.gen(on, PG);
  w = w / sum(w);                                    % weights proportional to Pg, sum to one
  Hw = makePTDF(mpc, w);
  dcol = max(abs(Hw - H), [], 1);                    % one number per bus column
  Pbus = makeSbus(mpc.baseMVA, mpc.bus, mpc.gen);    % net injections (pu), balanced in the DC model
  Pbus = real(Pbus) - sum(real(Pbus)) * w;           % remove the imbalance (losses) with the same weights
  dflow = max(abs(Hw * Pbus - H * Pbus)) * mpc.baseMVA;
end

%!test
%! if isempty(which('makePTDF')), disp('MATPOWER not on the path, test skipped'); return; end
%! [H, Hw, dcol, dflow] = ptdf_slack_case30('case30');
%! assert(size(H), [41 30]); assert(all(abs(H(:, 1)) < 1e-12))     % the slack column is zero
%! assert(max(dcol) > 0.01)                                        % the columns do change
%! assert(dflow < 1e-9)                                            % the flows of a balanced injection do not
