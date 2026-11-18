# Verdicts of the known-answer test of the path term (plan: tasks/todo.md, "PLAN (00:49, 18 Nov)").
# Input (jq -s): the by-day JSONs of the cases; inversions[0] = plain, [1] = with the term, [2] = second start with it.
def cm: (. * 100000 | round) / 1000;
def pooled: (map(. * .) | add / length) | sqrt;
def verdict(c): if c then "HOLDS" else "NOT MET" end;
map(if ((.inversions[0].dir | test("_p_n[0-9]+$")) and (.inversions[1].dir | test("_mp_n[0-9]+$"))) | not
    then error("order of the runs is not plain, with the term") else . end)
| . as $all
| (map(.inversions[1].vs_well.rmse_m - .inversions[0].vs_well.rmse_m)) as $d
| ($d | add / length) as $t1
| [ $all[] | .inversions[0] as $p | .inversions[1] as $m
    | range(0; ($p.sigma_by_day_m | length)) as $k
    | select($p.sigma_by_day_m[$k] <= 0.01)
    | {p: $p.vs_well_rms_by_day_m[$k], m: $m.vs_well_rms_by_day_m[$k]} ] as $D
| [ $all[] | select((.inversions | length) > 2)
    | .inversions[1].dir as $a | .inversions[2].dir as $b
    | if ($b | test("_mp_n2_s2$")) | not then error("third run is not the second start with the term") else . end
    | {w: (.data | sub(".*/twin_site_"; "")), rms: (.pairs[] | select(.a == $a and .b == $b) | .a_minus_b.rms_m)} ] as $S
| [ $all[] | .inversions[0] as $p | .inversions[1] as $m
    | range(0; ($p.sigma_by_day_m | length)) as $k
    | {p: ($p.vs_well_rms_by_day_m[$k] <= 2 * $p.sigma_by_day_m[$k]),
       m: ($m.vs_well_rms_by_day_m[$k] <= 2 * $m.sigma_by_day_m[$k])} ] as $C
| "cases read: \($all | length); d by case (cm): \($d | map(cm | tostring) | join(" ")); with the term nearer the truth in \($d | map(select(. < 0)) | length) of \($d | length)",
  "T1  mean of d over the cases: \($t1 | cm) cm (limit +0.1) -> \(verdict(($all | length) == 10 and $t1 <= 0.001))",
  ( if ($D | length) >= 10 then
      ($D | map(.p) | pooled) as $rp | ($D | map(.m) | pooled) as $rm
      | "T2  days with plain sd <= 1 cm: \($D | length) (case, day) pairs; pooled distance from the truth plain \($rp | cm) cm, with the term \($rm | cm) cm, difference \(($rm - $rp) | cm) cm (limit +0.1) -> \(verdict($rm <= $rp + 0.001))"
    else "T2  days with plain sd <= 1 cm: \($D | length) (case, day) pairs, fewer than 10 -> NOT MET" end ),
  "T3  two starts with the term, rms of the difference (cm): \($S | map("\(.w) \(.rms | cm)") | join("; ")) (limit 0.2 each, three windows) -> \(verdict(($S | length) == 3 and ($S | all(.rms <= 0.002))))",
  "coverage (reported): share of (case, day) pairs within 2 sd of the truth: plain \(($C | map(select(.p)) | length)) of \($C | length), with the term \(($C | map(select(.m)) | length)) of \($C | length) (\((($C | map(select(.m)) | length) / ($C | length) * 1000 | round) / 10) %)"
