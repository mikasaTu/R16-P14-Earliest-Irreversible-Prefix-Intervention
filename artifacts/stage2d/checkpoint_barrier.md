# Mandatory pre-atlas checkpoint barrier

Evidence commit: `9a1b98e2f6032e49fe2adfff1a9a6a646357c70c`
Evidence tree: `016f081ce8999f474d61292a86937b77b4f6b10c`
Branch isolation: **PASS**
Init-pool hash: `054c2fcfa71bc3f911a7c1b2eaf5709712ba535be4ae0b599d5a955cd8b0a46a`
Event availability: `{'put_the_cream_cheese_in_the_bowl': False, 'put_the_bowl_on_the_stove': True}`
Perturbation families: `{'put_the_cream_cheese_in_the_bowl': False, 'put_the_bowl_on_the_stove': False}`
Frozen parameters: `{"put_the_bowl_on_the_stove": {"parameters": [{"clearance_delta_m": 0.0, "diagnostic_fallback": true, "future_index": 9, "gate_distance": 0.2841269841269841, "parameter_id": "future_09__clearance_p000mm", "qualified": false}, {"clearance_delta_m": -0.01, "diagnostic_fallback": true, "future_index": 9, "gate_distance": 0.584126984126984, "parameter_id": "future_09__clearance_m010mm", "qualified": false}], "status": "BLOCKED"}, "put_the_cream_cheese_in_the_bowl": {"parameters": [{"diagnostic_fallback": false, "gate_distance": 0.0, "magnitude_m": 0.06, "parameter_id": "shift_060mm", "qualified": true}, {"diagnostic_fallback": true, "gate_distance": 0.16363636363636358, "magnitude_m": 0.08, "parameter_id": "shift_080mm", "qualified": false}], "status": "BLOCKED"}}`
Planned atlas volume: 10620 fresh-process branches
Blockers: `['BLOCKED_BY_EVENT_CONSTRUCTION', 'BLOCKED_BY_PERTURBATION_QUALIFICATION']`

If any blocker is present, the explicit user override permits the remaining matrix to run only as diagnostic evidence. It cannot change the failed gate or produce a positive/accepted label.

Next command: submit Stage2D atlas PAI phase from this committed barrier; preserve diagnostic_only=true if blockers is nonempty
