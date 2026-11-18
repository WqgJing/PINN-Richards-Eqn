# One block of lines per by-day JSON of the known-answer test (scripts/referee_step_check.py output).
# inversions[0] = plain, [1] = with the path term, [2] (if present) = with the term from the second start.
def cm: (. * 100000 | round) / 1000;
def cms: map(cm | tostring) | join(" ");
(.inversions[0].dir | test("_p_n[0-9]+$")) as $ok0
| (.inversions[1].dir | test("_mp_n[0-9]+$")) as $ok1
| if ($ok0 and $ok1) | not then error("order of the runs is not plain, with the term") else . end
| .inversions[0] as $p | .inversions[1] as $m
| "\($p.dir | sub(".*/twin_"; "") | sub("_p_n"; " seed "))",
  "  truth distance over the window (cm): plain \($p.vs_well.rmse_m | cm), with the term \($m.vs_well.rmse_m | cm), d = \(($m.vs_well.rmse_m - $p.vs_well.rmse_m) | cm)",
  "  start \($p.start) / \($m.start); cost \($p.cost) / \($m.cost); coefficients on the bound \($p.coef_on_bound | tostring) / \($m.coef_on_bound | tostring); worst-solve share not converged \($p.fem_not_converged_max) / \($m.fem_not_converged_max)",
  "  plain sd by day (cm):            \($p.sigma_by_day_m | cms)",
  "  with-term sd by day (cm):        \($m.sigma_by_day_m | cms)",
  "  plain distance by day (cm):      \($p.vs_well_rms_by_day_m | cms)",
  "  with-term distance by day (cm):  \($m.vs_well_rms_by_day_m | cms)",
  ( if (.inversions | length) > 2 then
      .inversions[2] as $b
      | (.pairs[] | select(.a == $m.dir and .b == $b.dir)) as $q
      | "  second start \($b.start) with the term: truth distance \($b.vs_well.rmse_m | cm) cm, cost \($b.cost), on the bound \($b.coef_on_bound | tostring); first minus second start: rms \($q.a_minus_b.rms_m | cm) cm, largest \($q.a_minus_b.maxabs_m | cm) cm"
    else empty end )
