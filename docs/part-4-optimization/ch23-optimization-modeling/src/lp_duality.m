function [x, z, lambda] = lp_duality()
  % The Lab LP of Chapter 23 in Octave: max 3 x1 + 5 x2 subject to x1 <= 4, 2 x2 <= 12, 3 x1 + 2 x2 <= 18, x >= 0,
  % solved with glpk. Returns the solution, the optimum and the dual values of the three constraints.
  c = [3; 5];
  A = [1 0; 0 2; 3 2];
  b = [4; 12; 18];
  lb = [0; 0];
  ub = [];
  ctype = 'UUU';          % three upper-bounded (<=) rows
  vartype = 'CC';         % continuous variables
  sense = -1;             % maximize
  [x, z, status, extra] = glpk(c, A, b, lb, ub, ctype, vartype, sense);
  if status ~= 0
    error('glpk did not solve the LP, status %d', status);
  end
  lambda = extra.lambda;  % dual values: the textbook shadow prices, positive for a maximization
end

%!test
%! [x, z, lambda] = lp_duality();
%! assert(z, 36, 1e-9)
%! assert(x, [2; 6], 1e-9)
%! assert(lambda, [0; 1.5; 1], 1e-9)             % y = (0, 1.5, 1)
%! assert([4 12 18] * lambda, z, 1e-9)           % strong duality: b . y = z
%! assert(lambda .* ([4; 12; 18] - [1 0; 0 2; 3 2] * x), zeros(3, 1), 1e-9)  % complementary slackness
