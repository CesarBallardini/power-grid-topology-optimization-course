function [v_nodal, v_mesh, i_mesh] = mesh_nodal()
  % The worked example of Chapter 8 solved twice: by nodal analysis (G v = i) and by mesh analysis.
  % Conductances 1 S (node 1 to ground), 2 S (nodes 1-2), 4 S (node 2 to ground); a 3 A source feeds node 1.
  % Mesh 1 holds the source and the 1 ohm resistor, so I1 = 3 A; mesh 2 runs through the 1, 1/2 and 1/4 ohm
  % resistors: KVL gives Ra (I2 - I1) + (Rb + Rc) I2 = 0.
  G = [1 + 2, -2; -2, 2 + 4];
  v_nodal = G \ [3; 0];
  Ra = 1; Rb = 1 / 2; Rc = 1 / 4; Is = 3;
  R = [1, 0; -Ra, Ra + Rb + Rc];
  i_mesh = R \ [Is; 0];
  v_mesh = [Ra * (i_mesh(1) - i_mesh(2)); Rc * i_mesh(2)];
end

%!test
%! [v_nodal, v_mesh, i_mesh] = mesh_nodal();
%! assert(v_nodal, [18 / 14; 6 / 14], 1e-12)
%! assert(v_mesh, v_nodal, 1e-12)
%! assert(i_mesh, [3; 12 / 7], 1e-12)      % 3 A in the source mesh, 1.7143 A through the 2 S and 4 S branches
%! assert(i_mesh(1) - i_mesh(2), v_nodal(1) * 1, 1e-12)   % current in the 1 S branch
