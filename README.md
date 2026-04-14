# NoBSC: Notears-Based Smooth Constraint
Directed Acyclic Graph (DAG) causal structural learning seeks to uncover
the true causal relationships among observed variables by representing them
as directed graphs without cycles. Many existing methods, however, cannot
easily incorporate domain knowledge, particularly forbidden-edge constraints,
which specify certain edges should not exist. Ignoring these constraints can in-
troduce spurious associations, as some variables may be determined externally
and cannot be influenced by others; any incoming edges to such variables are
invalid and must be excluded. Recent algorithms have attempted to integrate
prior knowledge by adding constraints via continuous optimization. While
this approach can improve the ability of incorporating known forbidden edges
during DAG estimation, the functional form of the structural constraints in-
cludes non-smooth components. This lack of smoothness limits theoretical
guarantees for standard gradient-based optimization methods and can lead
to inconsistencies within a continuous optimization framework. To address
these challenges, we propose a structural learning algorithm that directly
incorporates gradient information associated with forbidden-edge constraints.
This reformulation preserves the differentiability of the structural constraint
function, thereby enhancing the numerical stability of gradient-based op-
timization while softly enforcing domain-specific exclusions. Simulation
studies on both synthetic and real-world datasets demonstrate that our method
effectively eliminates forbidden edges, maintains sparsity, and achieves more
accurate recovery of the true causal structure compared to existing benchmark
approaches.
