# Code-first mechanism reverse audit

This audit explains the preregistered increases and decreases; it does not generate a new idea.
All rows are diagnostic-only because event construction and perturbation qualification are upstream blockers; formal_positive_evidence_allowed=false.
Method-error rows and replay-nonadmitted rows are excluded from the causal contrasts.

## Rule distribution and action-content result

Frozen rule_k distribution overall: {'5': 1, '7': 82, '10': 7, '11': 64}.
Frozen rule_k distribution by task: {'put_the_bowl_on_the_stove': {'7': 82}, 'put_the_cream_cheese_in_the_bowl': {'5': 1, '10': 7, '11': 64}}.
Mean cached-vs-fresh action disagreement: {'overall': {'n': 154, 'mean': 0.09399220906231621}, 'put_the_bowl_on_the_stove': {'n': 82, 'mean': 0.08371746410593027}, 'put_the_cream_cheese_in_the_bowl': {'n': 72, 'mean': 0.10569400192931133}}.
The disagreement confirms that cached and fresh prefixes are not byte-identical, but the matched-k safe-success difference is zero and no real efficiency field reaches 15%; this is action-content difference without demonstrated outcome value.

## Mechanistic constraints and H4 interpretation

These statements bind the interpretation to the implemented arms and the observed rows; they do not introduce a new selector.

- H2 code constraint: CACHED and FRESH execute equal prefix lengths, matched detection-time calls, the same tail actor, and h_tail=4; only old[d:k] versus fresh_d[:k-d] differs. Observed safe-success delta is 0, action disagreement is nonzero, and no real efficiency field reaches 15%.
- H3 code constraint: IMMEDIATE_FRESH sets k=d and executes only the four-step common tail with one call, whereas EVENT_ALIGNED_CACHED executes k-d cached actions plus the same tail and an additional call. This structural exposure produces +6.753 actions and +0.266 cause overall; cream (mostly k=11) is +8.82 actions/+0.569 cause, bowl (k=7) is +4.94 actions/zero cause delta.
- H4 code constraint: FIXED_DELAY_8 always uses k=10, while the frozen event rule uses the preregistered release/path formula. The event rule is -0.0065 safe success, +0.0844 cause, and -1.182 actions (9.9% reduction, below 15%); calibration-only oracle gap recovery is below 40%. Bowl saves about 2.94 actions at no cause cost, while cream is mostly k=11 and costs about 0.82 actions with +18.06 percentage points cause.
- Sham-query code constraint: CACHED_MATCHED calls ACT at d and discards the output; CACHED_NOQUERY omits it but executes identical action bytes. The exact signatures isolate +1 call/+0.0051 s as computation-only overhead.

## cached action content under matched handoff time

execute_branch selects old[d:k] for CACHED_MATCHED and fresh_d[:k-d] for FRESH_MATCHED, then both call the same actor at k and execute h_tail=4

The matched-k contrast has safe-success delta 0.0 and cause-violation delta -0.012987012987012988.  In this matrix safe success is effectively unchanged, so the old cached action content does not show a measurable H2 advantage over a fresh prefix at the same handoff.  Any small path/progress difference is motion geometry, not a selection-quality gain; call count, prefix length, tail actor, and tail horizon are matched.

Observed mean effects (left minus right):

- safe_success: 0.0 (equal)
- cause_violation: -0.012987012987012988 (lower_in_left)
- actual_post_detection_actions: 0.006493506493506494 (higher_in_left)
- actual_actor_calls: 0.0 (equal)
- actual_inference_wall_time_s: 6.870211692543095e-05 (higher_in_left)
- eef_path_length_m: -0.0008340763917824927 (lower_in_left)
- manipulated_object_path_length_m: -0.0008893863803233139 (lower_in_left)
- progress_regression_m: -0.000810471871250343 (lower_in_left)

Stratified diagnostic effects (task/severity/actor seed; not pooled as population samples):

### task
- put_the_bowl_on_the_stove (n=82): safe_success=0.0 (equal); cause_violation=0.0 (equal); actual_post_detection_actions=0.012195121951219513 (higher_in_left); eef_path_length_m=-0.0016030309041241816 (lower_in_left); progress_regression_m=-0.0013630536262453582 (lower_in_left)
- put_the_cream_cheese_in_the_bowl (n=72): safe_success=0.0 (equal); cause_violation=-0.027777777777777776 (lower_in_left); actual_post_detection_actions=0.0 (equal); eef_path_length_m=4.167735838443073e-05 (higher_in_left); progress_regression_m=-0.00018114265028379775 (lower_in_left)

### severity
- put_the_bowl_on_the_stove::future_09__clearance_m010mm (n=40): safe_success=0.0 (equal); cause_violation=0.0 (equal); actual_post_detection_actions=0.025 (higher_in_left); eef_path_length_m=-0.0023428918622714987 (lower_in_left); progress_regression_m=-0.0019702296561762106 (lower_in_left)
- put_the_bowl_on_the_stove::future_09__clearance_p000mm (n=42): safe_success=0.0 (equal); cause_violation=0.0 (equal); actual_post_detection_actions=0.0 (equal); eef_path_length_m=-0.0008984014201743553 (lower_in_left); progress_regression_m=-0.0007847907405969274 (lower_in_left)
- put_the_cream_cheese_in_the_bowl::shift_060mm (n=37): safe_success=0.0 (equal); cause_violation=0.0 (equal); actual_post_detection_actions=0.0 (equal); eef_path_length_m=0.00041135043252740793 (higher_in_left); progress_regression_m=-0.00034686564148521507 (lower_in_left)
- put_the_cream_cheese_in_the_bowl::shift_080mm (n=35): safe_success=0.0 (equal); cause_violation=-0.05714285714285714 (lower_in_left); actual_post_detection_actions=0.0 (equal); eef_path_length_m=-0.0003491198914238594 (lower_in_left); progress_regression_m=-5.949773870870891e-06 (lower_in_left)

### actor_seed
- 17 (n=64): safe_success=0.0 (equal); cause_violation=-0.03125 (lower_in_left); actual_post_detection_actions=0.0 (equal); eef_path_length_m=0.0004641126412089165 (higher_in_left); progress_regression_m=0.0007329949598190938 (higher_in_left)
- 29 (n=47): safe_success=0.0 (equal); cause_violation=-0.02127659574468085 (lower_in_left); actual_post_detection_actions=0.0 (equal); eef_path_length_m=-0.0027613366290212536 (lower_in_left); progress_regression_m=-0.003340502750537849 (lower_in_left)
- 7 (n=43): safe_success=0.0 (equal); cause_violation=0.023255813953488372 (higher_in_left); actual_post_detection_actions=0.023255813953488372 (higher_in_left); eef_path_length_m=-0.0006597244606482696 (lower_in_left); progress_regression_m=-0.00034234224013246365 (lower_in_left)

## detection-time sham query overhead

CACHED_MATCHED calls ACT at d and discards the tensor; CACHED_NOQUERY executes identical old[d:k] bytes

Exact pre-tail signatures show the discarded query has no physical or history effect; the observed actor-call delta is 1.0 and the inference-time delta is 0.005122639487006842.  Thus the sham-query increase is pure computation/bookkeeping overhead, not action-content value.

Observed mean effects (left minus right):

- safe_success: 0.0 (equal)
- cause_violation: 0.0 (equal)
- actual_post_detection_actions: 0.0 (equal)
- actual_actor_calls: 1.0 (higher_in_left)
- actual_inference_wall_time_s: 0.005122639487006842 (higher_in_left)
- eef_path_length_m: 0.0 (equal)
- manipulated_object_path_length_m: 0.0 (equal)
- progress_regression_m: 0.0 (equal)

Stratified diagnostic effects (task/severity/actor seed; not pooled as population samples):

### task
- put_the_bowl_on_the_stove (n=82): safe_success=0.0 (equal); cause_violation=0.0 (equal); actual_post_detection_actions=0.0 (equal); eef_path_length_m=0.0 (equal); progress_regression_m=0.0 (equal)
- put_the_cream_cheese_in_the_bowl (n=72): safe_success=0.0 (equal); cause_violation=0.0 (equal); actual_post_detection_actions=0.0 (equal); eef_path_length_m=0.0 (equal); progress_regression_m=0.0 (equal)

### severity
- put_the_bowl_on_the_stove::future_09__clearance_m010mm (n=40): safe_success=0.0 (equal); cause_violation=0.0 (equal); actual_post_detection_actions=0.0 (equal); eef_path_length_m=0.0 (equal); progress_regression_m=0.0 (equal)
- put_the_bowl_on_the_stove::future_09__clearance_p000mm (n=42): safe_success=0.0 (equal); cause_violation=0.0 (equal); actual_post_detection_actions=0.0 (equal); eef_path_length_m=0.0 (equal); progress_regression_m=0.0 (equal)
- put_the_cream_cheese_in_the_bowl::shift_060mm (n=37): safe_success=0.0 (equal); cause_violation=0.0 (equal); actual_post_detection_actions=0.0 (equal); eef_path_length_m=0.0 (equal); progress_regression_m=0.0 (equal)
- put_the_cream_cheese_in_the_bowl::shift_080mm (n=35): safe_success=0.0 (equal); cause_violation=0.0 (equal); actual_post_detection_actions=0.0 (equal); eef_path_length_m=0.0 (equal); progress_regression_m=0.0 (equal)

### actor_seed
- 17 (n=64): safe_success=0.0 (equal); cause_violation=0.0 (equal); actual_post_detection_actions=0.0 (equal); eef_path_length_m=0.0 (equal); progress_regression_m=0.0 (equal)
- 29 (n=47): safe_success=0.0 (equal); cause_violation=0.0 (equal); actual_post_detection_actions=0.0 (equal); eef_path_length_m=0.0 (equal); progress_regression_m=0.0 (equal)
- 7 (n=43): safe_success=0.0 (equal); cause_violation=0.0 (equal); actual_post_detection_actions=0.0 (equal); eef_path_length_m=0.0 (equal); progress_regression_m=0.0 (equal)

## event-aligned delay versus immediate replanning

arm_plan maps IMMEDIATE_FRESH to k=d and EVENT_ALIGNED_CACHED to the frozen event rule

Against immediate replanning, the event-aligned arm adds 6.753 post-detection actions and changes cause violation by 0.266; its safe-success delta is only 0.013.  This identifies stale-action exposure during the delayed window rather than a free timing benefit.  Execution-cost changes use only measured actions, calls, paths, and wall time; retained cached count is not efficiency.

Observed mean effects (left minus right):

- safe_success: 0.012987012987012988 (higher_in_left)
- cause_violation: 0.2662337662337662 (higher_in_left)
- actual_post_detection_actions: 6.753246753246753 (higher_in_left)
- actual_actor_calls: 1.0 (higher_in_left)
- actual_inference_wall_time_s: 0.005321890701301905 (higher_in_left)
- eef_path_length_m: 0.057106601413889374 (higher_in_left)
- manipulated_object_path_length_m: 0.05623575740898827 (higher_in_left)
- progress_regression_m: 0.0016915665108350453 (higher_in_left)

Stratified diagnostic effects (task/severity/actor seed; not pooled as population samples):

### task
- put_the_bowl_on_the_stove (n=82): safe_success=0.024390243902439025 (higher_in_left); cause_violation=0.0 (equal); actual_post_detection_actions=4.939024390243903 (higher_in_left); eef_path_length_m=0.05450985981838351 (higher_in_left); progress_regression_m=-6.337841794737597e-05 (lower_in_left)
- put_the_cream_cheese_in_the_bowl (n=72): safe_success=0.0 (equal); cause_violation=0.5694444444444444 (higher_in_left); actual_post_detection_actions=8.819444444444445 (higher_in_left); eef_path_length_m=0.06006400156432661 (higher_in_left); progress_regression_m=0.0036902537908372474 (higher_in_left)

### severity
- put_the_bowl_on_the_stove::future_09__clearance_m010mm (n=40): safe_success=0.05 (higher_in_left); cause_violation=0.0 (equal); actual_post_detection_actions=4.875 (higher_in_left); eef_path_length_m=0.0522552422762928 (higher_in_left); progress_regression_m=0.00035253873432244563 (higher_in_left)
- put_the_bowl_on_the_stove::future_09__clearance_p000mm (n=42): safe_success=0.0 (equal); cause_violation=0.0 (equal); actual_post_detection_actions=5.0 (higher_in_left); eef_path_length_m=0.05665711462037465 (higher_in_left); progress_regression_m=-0.00045948999153768225 (lower_in_left)
- put_the_cream_cheese_in_the_bowl::shift_060mm (n=37): safe_success=0.0 (equal); cause_violation=0.5135135135135135 (higher_in_left); actual_post_detection_actions=8.891891891891891 (higher_in_left); eef_path_length_m=0.05823604992272712 (higher_in_left); progress_regression_m=0.005696563935602438 (higher_in_left)
- put_the_cream_cheese_in_the_bowl::shift_080mm (n=35): safe_success=0.0 (equal); cause_violation=0.6285714285714286 (higher_in_left); actual_post_detection_actions=8.742857142857142 (higher_in_left); eef_path_length_m=0.061996407585446035 (higher_in_left); progress_regression_m=0.001569297352085473 (higher_in_left)

### actor_seed
- 17 (n=64): safe_success=0.0 (equal); cause_violation=0.265625 (higher_in_left); actual_post_detection_actions=6.6875 (higher_in_left); eef_path_length_m=0.058471698101223844 (higher_in_left); progress_regression_m=0.003128734015251093 (higher_in_left)
- 29 (n=47): safe_success=0.0 (equal); cause_violation=0.3404255319148936 (higher_in_left); actual_post_detection_actions=7.531914893617022 (higher_in_left); eef_path_length_m=0.05580695938626909 (higher_in_left); progress_regression_m=0.0012552084596307886 (higher_in_left)
- 7 (n=43): safe_success=0.046511627906976744 (higher_in_left); cause_violation=0.18604651162790697 (higher_in_left); actual_post_detection_actions=6.0 (higher_in_left); eef_path_length_m=0.05649536856060441 (higher_in_left); progress_regression_m=2.9476002090231027e-05 (higher_in_left)

## event-aligned rule versus strongest fixed delay

The frozen rule uses target release/path timing and selects rule_k; FIXED_DELAY_8 always executes prefix_k=d+8=10 before the same h_tail=4 recovery call.

Against FIXED_DELAY_8, the event rule changes safe success by -0.0065 (-0.65 percentage points), cause violation by 0.0844 (+8.44 percentage points), and post-detection actions by -1.182 (-1.18, only about 9.9% of the 11.873-action fixed baseline).  The bowl rule is k=7 throughout: it saves about 2.94 actions with no cause difference.  Cream is mostly k=11: it spends about +0.82 actions and has +18.06 percentage points cause violation. The calibration-only oracle gap recovery is 0.158 when available, below the 0.40 H4 criterion. Thus the fixed-baseline comparison is heterogeneous and does not support H4; it is an audit of the frozen rule, not a new method.

Observed mean effects (left minus right):

- safe_success: -0.006493506493506494 (lower_in_left)
- cause_violation: 0.08441558441558442 (higher_in_left)
- actual_post_detection_actions: -1.1818181818181819 (lower_in_left)
- actual_actor_calls: 0.012987012987012988 (higher_in_left)
- actual_inference_wall_time_s: 8.030120782872732e-05 (higher_in_left)
- eef_path_length_m: -0.015569539131781458 (lower_in_left)
- manipulated_object_path_length_m: -0.016035010756945536 (lower_in_left)
- progress_regression_m: 0.0007188263503113188 (higher_in_left)

Stratified diagnostic effects (task/severity/actor seed; not pooled as population samples):

### task
- put_the_bowl_on_the_stove (n=82): safe_success=-0.012195121951219513 (lower_in_left); cause_violation=0.0 (equal); actual_post_detection_actions=-2.9390243902439024 (lower_in_left); eef_path_length_m=-0.031145570738649674 (lower_in_left); progress_regression_m=0.0010806835450924799 (higher_in_left)
- put_the_cream_cheese_in_the_bowl (n=72): safe_success=0.0 (equal); cause_violation=0.18055555555555555 (higher_in_left); actual_post_detection_actions=0.8194444444444444 (higher_in_left); eef_path_length_m=0.002169830198262902 (higher_in_left); progress_regression_m=0.00030671121181055194 (higher_in_left)

### severity
- put_the_bowl_on_the_stove::future_09__clearance_m010mm (n=40): safe_success=-0.025 (lower_in_left); cause_violation=0.0 (equal); actual_post_detection_actions=-2.875 (lower_in_left); eef_path_length_m=-0.02983603097752029 (lower_in_left); progress_regression_m=0.001918052394691372 (higher_in_left)
- put_the_bowl_on_the_stove::future_09__clearance_p000mm (n=42): safe_success=0.0 (equal); cause_violation=0.0 (equal); actual_post_detection_actions=-3.0 (lower_in_left); eef_path_length_m=-0.032392751463534805 (lower_in_left); progress_regression_m=0.00028318940261734456 (higher_in_left)
- put_the_cream_cheese_in_the_bowl::shift_060mm (n=37): safe_success=0.0 (equal); cause_violation=0.21621621621621623 (higher_in_left); actual_post_detection_actions=0.8918918918918919 (higher_in_left); eef_path_length_m=0.0023550265780755133 (higher_in_left); progress_regression_m=0.00042749308701522075 (higher_in_left)
- put_the_cream_cheese_in_the_bowl::shift_080mm (n=35): safe_success=0.0 (equal); cause_violation=0.14285714285714285 (higher_in_left); actual_post_detection_actions=0.7428571428571429 (higher_in_left); eef_path_length_m=0.001974051168175285 (higher_in_left); progress_regression_m=0.00017902751516561636 (higher_in_left)

### actor_seed
- 17 (n=64): safe_success=0.0 (equal); cause_violation=0.09375 (higher_in_left); actual_post_detection_actions=-1.3125 (lower_in_left); eef_path_length_m=-0.01743306840473459 (lower_in_left); progress_regression_m=0.0003062399056844721 (higher_in_left)
- 29 (n=47): safe_success=-0.02127659574468085 (lower_in_left); cause_violation=0.0851063829787234 (higher_in_left); actual_post_detection_actions=-0.425531914893617 (lower_in_left); eef_path_length_m=-0.007605408871237325 (lower_in_left); progress_regression_m=0.001839179744113134 (higher_in_left)
- 7 (n=43): safe_success=0.0 (equal); cause_violation=0.06976744186046512 (higher_in_left); actual_post_detection_actions=-1.813953488372093 (lower_in_left); eef_path_length_m=-0.02150089375449248 (lower_in_left); progress_regression_m=0.00010833618629812959 (higher_in_left)

## motion content versus elapsed-time control

HOLD_MATCHED preserves gripper state but zeros six motion dimensions for exactly k-d actions

The contrast separates old motion content from merely waiting the same number of simulator steps: cached versus hold changes cause violation by 0.078 and EEF path by 0.079 m.  This is evidence of a physical motion-content difference in the diagnostic replay, not deployable value.

Observed mean effects (left minus right):

- safe_success: 0.0 (equal)
- cause_violation: 0.07792207792207792 (higher_in_left)
- actual_post_detection_actions: -0.032467532467532464 (lower_in_left)
- actual_actor_calls: 0.0 (equal)
- actual_inference_wall_time_s: 6.213104542846898e-05 (higher_in_left)
- eef_path_length_m: 0.07934518094070642 (higher_in_left)
- manipulated_object_path_length_m: 0.07062086432562284 (higher_in_left)
- progress_regression_m: 0.000572092970158946 (higher_in_left)

Stratified diagnostic effects (task/severity/actor seed; not pooled as population samples):

### task
- put_the_bowl_on_the_stove (n=82): safe_success=0.0 (equal); cause_violation=0.0 (equal); actual_post_detection_actions=-0.06097560975609756 (lower_in_left); eef_path_length_m=0.08059237060153632 (higher_in_left); progress_regression_m=0.0012617876568999755 (higher_in_left)
- put_the_cream_cheese_in_the_bowl (n=72): safe_success=0.0 (equal); cause_violation=0.16666666666666666 (higher_in_left); actual_post_detection_actions=0.0 (equal); eef_path_length_m=0.07792477049365013 (higher_in_left); progress_regression_m=-0.00021339264529611528 (lower_in_left)

### severity
- put_the_bowl_on_the_stove::future_09__clearance_m010mm (n=40): safe_success=0.0 (equal); cause_violation=0.0 (equal); actual_post_detection_actions=-0.125 (lower_in_left); eef_path_length_m=0.07772809720896083 (higher_in_left); progress_regression_m=0.0026184569587461197 (higher_in_left)
- put_the_bowl_on_the_stove::future_09__clearance_p000mm (n=42): safe_success=0.0 (equal); cause_violation=0.0 (equal); actual_post_detection_actions=0.0 (equal); eef_path_length_m=0.08332025002303675 (higher_in_left); progress_regression_m=-3.027834485825718e-05 (lower_in_left)
- put_the_cream_cheese_in_the_bowl::shift_060mm (n=37): safe_success=0.0 (equal); cause_violation=0.13513513513513514 (higher_in_left); actual_post_detection_actions=0.0 (equal); eef_path_length_m=0.07514178340283582 (higher_in_left); progress_regression_m=-5.8758760681842936e-05 (lower_in_left)
- put_the_cream_cheese_in_the_bowl::shift_080mm (n=35): safe_success=0.0 (equal); cause_violation=0.2 (higher_in_left); actual_post_detection_actions=0.0 (equal); eef_path_length_m=0.08086678541822523 (higher_in_left); progress_regression_m=-0.00037686275188834645 (lower_in_left)

### actor_seed
- 17 (n=64): safe_success=0.0 (equal); cause_violation=-0.03125 (lower_in_left); actual_post_detection_actions=0.0 (equal); eef_path_length_m=0.08208723652719775 (higher_in_left); progress_regression_m=0.0027383519568877582 (higher_in_left)
- 29 (n=47): safe_success=0.0 (equal); cause_violation=0.2127659574468085 (higher_in_left); actual_post_detection_actions=0.0 (equal); eef_path_length_m=0.07600072552688669 (higher_in_left); progress_regression_m=-0.00234460448341225 (lower_in_left)
- 7 (n=43): safe_success=0.0 (equal); cause_violation=0.09302325581395349 (higher_in_left); actual_post_detection_actions=-0.11627906976744186 (lower_in_left); eef_path_length_m=0.078919549473592 (higher_in_left); progress_regression_m=0.0005359116949776026 (higher_in_left)

## bounded reuse versus stale full-chunk execution

FULL_OLD_CHUNK executes old[d:H] with no recovery tail; the event rule truncates reuse and invokes the actor at k

Relative to the full stale chunk, bounded reuse changes cause violation by -0.136 and safe success by -0.026, while using -3.247 fewer post-detection actions.  The trade-off is consistent with truncating stale execution, but does not establish a positive selector because the upstream gate failed.

Observed mean effects (left minus right):

- safe_success: -0.025974025974025976 (lower_in_left)
- cause_violation: -0.13636363636363635 (lower_in_left)
- actual_post_detection_actions: -3.2467532467532467 (lower_in_left)
- actual_actor_calls: 2.0 (higher_in_left)
- actual_inference_wall_time_s: 0.010403555558443625 (higher_in_left)
- eef_path_length_m: -0.029719526269599863 (lower_in_left)
- manipulated_object_path_length_m: -0.030962627251537717 (lower_in_left)
- progress_regression_m: 0.00016623759845956553 (higher_in_left)

Stratified diagnostic effects (task/severity/actor seed; not pooled as population samples):

### task
- put_the_bowl_on_the_stove (n=82): safe_success=-0.04878048780487805 (lower_in_left); cause_violation=0.0 (equal); actual_post_detection_actions=-5.060975609756097 (lower_in_left); eef_path_length_m=-0.05381715568770314 (lower_in_left); progress_regression_m=0.0010732966557642756 (higher_in_left)
- put_the_cream_cheese_in_the_bowl (n=72): safe_success=0.0 (equal); cause_violation=-0.2916666666666667 (lower_in_left); actual_post_detection_actions=-1.1805555555555556 (lower_in_left); eef_path_length_m=-0.002275003876760018 (lower_in_left); progress_regression_m=-0.0008668018834707986 (lower_in_left)

### severity
- put_the_bowl_on_the_stove::future_09__clearance_m010mm (n=40): safe_success=-0.025 (lower_in_left); cause_violation=0.0 (equal); actual_post_detection_actions=-5.125 (lower_in_left); eef_path_length_m=-0.05166658107998895 (lower_in_left); progress_regression_m=0.0019029092715685532 (higher_in_left)
- put_the_bowl_on_the_stove::future_09__clearance_p000mm (n=42): safe_success=-0.07142857142857142 (lower_in_left); cause_violation=0.0 (equal); actual_post_detection_actions=-5.0 (lower_in_left); eef_path_length_m=-0.055865321980764306 (lower_in_left); progress_regression_m=0.00028318940261734456 (higher_in_left)
- put_the_cream_cheese_in_the_bowl::shift_060mm (n=37): safe_success=0.0 (equal); cause_violation=-0.2702702702702703 (lower_in_left); actual_post_detection_actions=-1.1081081081081081 (lower_in_left); eef_path_length_m=-0.0024620803622601604 (lower_in_left); progress_regression_m=-0.0012323071873929436 (lower_in_left)
- put_the_cream_cheese_in_the_bowl::shift_080mm (n=35): safe_success=0.0 (equal); cause_violation=-0.3142857142857143 (lower_in_left); actual_post_detection_actions=-1.2571428571428571 (lower_in_left); eef_path_length_m=-0.0020772373063741535 (lower_in_left); progress_regression_m=-0.000480410562181674 (lower_in_left)

### actor_seed
- 17 (n=64): safe_success=-0.015625 (lower_in_left); cause_violation=-0.15625 (lower_in_left); actual_post_detection_actions=-3.3125 (lower_in_left); eef_path_length_m=-0.032899276457269845 (lower_in_left); progress_regression_m=-0.0006943670430824061 (lower_in_left)
- 29 (n=47): safe_success=-0.02127659574468085 (lower_in_left); cause_violation=-0.2127659574468085 (lower_in_left); actual_post_detection_actions=-2.4680851063829787 (lower_in_left); eef_path_length_m=-0.01773210030739205 (lower_in_left); progress_regression_m=0.0015148893823392704 (higher_in_left)
- 7 (n=43): safe_success=-0.046511627906976744 (lower_in_left); cause_violation=-0.023255813953488372 (lower_in_left); actual_post_detection_actions=-4.0 (lower_in_left); eef_path_length_m=-0.03808941018152752 (lower_in_left); progress_regression_m=-2.6970233718572786e-05 (lower_in_left)
