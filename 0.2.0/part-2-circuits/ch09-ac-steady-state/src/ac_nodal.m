function [V, Iinj, Ib, Sin, Sabs] = ac_nodal(fr, to, z, V1, I2)
  % Complex nodal analysis of the three-node circuit of Chapter 9, Exercise 5: node 1 is ground, node 2
  % is held at V1 by a voltage source, node 3 receives the current I2; branches have impedances z between
  % nodes fr and to (indices from 1). Returns the node voltages, the net injections Y * V (the second one
  % is the current of the voltage source), the branch currents and the injected and absorbed complex power.
  m = numel(fr);
  A = zeros(m, 3);
  A(sub2ind([m 3], 1:m, fr)) = 1;
  A(sub2ind([m 3], 1:m, to)) = -1;
  Y = A' * diag(1 ./ z) * A;
  V = [0; V1; (I2 - Y(3, 2) * V1) / Y(3, 3)];
  Iinj = Y * V;
  Ib = (V(fr) - V(to)) ./ z(:);
  Sin = sum(V(2:3) .* conj(Iinj(2:3)));
  Sabs = sum(abs(Ib).^2 .* z(:));
end

%!test
%! [V, Iinj, Ib, Sin, Sabs] = ac_nodal([2 3], [3 1], [2 + 5i, 10 - 4i], 230, 5 * exp(-1i * pi / 6));
%! assert(V(1), 0)
%! assert(V(2), 230)
%! assert(Sin, Sabs, 1e-9)                              % Tellegen: injected equals absorbed
%! assert(abs(Ib(1)), abs(Iinj(2)), 1e-9)               % the voltage source current flows through RL
%! assert(Iinj(1) + Iinj(2) + Iinj(3), 0, 1e-9)         % KCL over the whole circuit
