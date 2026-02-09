# Human-in-the-Loop Evaluation

After training, the best policy is evaluated against (a) held-out simulated
trajectories and (b) a CLI-based rater interface where a human rates each
policy recommendation on a 1-5 Likert scale.

Rater sessions are logged to JSON in `results/human_eval/`. Aggregate scores
are compared to the simulated evaluation; divergence between the two is
itself a finding worth inspecting.
