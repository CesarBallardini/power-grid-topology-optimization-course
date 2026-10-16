function [x, fval, info, output] = fsolve_example(x0)
  % Solve the worked example of Chapter 6 (x^2 + y^2 = 4, exp(x) + y = 1) with Octave's fsolve from x0
  % and return its output structure, whose field 'iterations' is compared with the Newton code.
  F = @(v) [v(1)^2 + v(2)^2 - 4; exp(v(1)) + v(2) - 1];
  opts = optimset('TolFun', 1e-12, 'TolX', 1e-12);
  [x, fval, info, output] = fsolve(F, x0(:), opts);
end

%!test
%! [x, fval, info, output] = fsolve_example([1; -1]);
%! assert(x, [1.004169; -1.729637], 5e-6)
%! assert(norm(fval) < 1e-8)
%! assert(info, 1)
%! assert(isfield(output, 'iterations') && output.iterations >= 3)
%!test
%! x = fsolve_example([-1; 1]);
%! assert(x, [-1.816264; 0.837368], 5e-6)
