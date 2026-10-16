function rho = perf_profile(T, taus)
  % Dolan-More performance profile (Chapter 26, Exercise 3).
  % T(p, s) is the time solver s needed on instance p, NaN (or Inf) for a failure; taus is a vector of
  % thresholds. rho(i, s) is the fraction of instances whose performance ratio T(p, s) / min_s T(p, s) is at
  % most taus(i). Failures get an infinite ratio and never count.
  T(isnan(T)) = Inf;
  best = min(T, [], 2);
  r = T ./ best;                       % performance ratios, Inf where the solver failed
  rho = zeros(numel(taus), size(T, 2));
  for i = 1:numel(taus)
    rho(i, :) = mean(isfinite(r) & r <= taus(i), 1);   % failures never count, not even for tau = Inf
  end
end

%!test
%! T = [1 2 NaN; 2 1 4; 3 3 1; 1 Inf 2];
%! rho = perf_profile(T, [1 2 Inf]);
%! assert(rho(1, :), [0.5 0.25 0.25], 1e-12)     % rho_s(1): fraction of instances where s was fastest
%! assert(rho(2, :), [0.75 0.5 0.5], 1e-12)      % within a factor 2 of the best
%! assert(rho(3, :), [1 0.75 0.75], 1e-12)       % large tau: fraction of instances solved at all
%!test
%! rho = perf_profile([1 1; 1 1], [1 3]);        % ties: every solver is "fastest" on every instance
%! assert(rho, [1 1; 1 1])
