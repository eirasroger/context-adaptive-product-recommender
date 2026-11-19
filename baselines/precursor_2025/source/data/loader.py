import json
import numpy as np
from .config import STAKEHOLDER_PREFS, SCENARIO_PREFS, PERFORMANCE_INDICATOR_RELEVANCE_MAPPING as PERF_MAP

# --- encoders for stakeholder & scenario prefs (one-hot over predefined lists) ---
def encode_stakeholder_pref(pref_list, all_possible_prefs=STAKEHOLDER_PREFS):
    vec = np.zeros(len(all_possible_prefs), dtype=np.float32)
    for p in pref_list:
        try:
            idx = all_possible_prefs.index(p)
            vec[idx] = 1.0
        except ValueError:
            pass
    return vec

def encode_scenario_pref(pref_list, all_possible_prefs=SCENARIO_PREFS):
    vec = np.zeros(len(all_possible_prefs), dtype=np.float32)
    for p in pref_list:
        try:
            idx = all_possible_prefs.index(p)
            vec[idx] = 1.0
        except ValueError:
            pass
    return vec

# --- build performance indicator universe (for relevance checks) ---
_PERF_INDICATORS_UNIVERSE = set()
for v in PERF_MAP.values():
    _PERF_INDICATORS_UNIVERSE.update(v)

def _extract_features_from_alt(alt, feature_keys, relevant_indicators):
    """
    Produce triplets for every feature: [value, present_flag, relevant_flag]
    present_flag: 1 if numeric value present, else 0
    relevant_flag: 1 if indicator is relevant for the scenario (from situations), else 0
    The cost 'c' is aggregated and appended as a final triplet.
    """
    feats = []

    for k in feature_keys:
        # Determine relevance: only indicators in the PERF universe are conditional.
        if k in _PERF_INDICATORS_UNIVERSE:
            is_relevant = 1 if k in relevant_indicators else 0
        else:
            # Non-performance indicators considered universally relevant
            is_relevant = 1

        v = alt.get(k, None)
        if isinstance(v, (int, float)):
            feats.append(float(v))      # value
            feats.append(1)             # present
            feats.append(is_relevant)   # relevant
        else:
            feats.append(0.0)           # value placeholder
            feats.append(0)             # present = 0
            feats.append(is_relevant)   # relevant may be 0 or 1

    # Aggregate numeric cost components into single cost feature (append as triplet)
    cost_values = alt.get('c', {}).values() if isinstance(alt.get('c', {}), dict) else []
    numeric_costs = [v for v in cost_values if isinstance(v, (int, float))]
    if numeric_costs:
        feats.append(float(sum(numeric_costs)))
        feats.append(1)
        feats.append(1)   # treat cost as relevant
    else:
        feats.append(0.0)
        feats.append(0)
        feats.append(1)   # cost relevance = 1 

    return feats


def load_data(frozen_path, labeled_path):
    """
    Loads frozen scenarios and labeled prefs and returns a list of scenario dicts:
      { 'id': sid,
        'features': np.array(shape=(n_alts, feat_dim)),
        'prefs': [float|None,...],
        'confs':  [float|None,...],
        'feature_keys': [...] }

    Encoding:
      For N numeric indicators (+cost), the per-indicator encoding is triplets (value,present,relevant).
      Final model input per alternative is: [ value1, present1, relevant1, ..., valueN, presentN, relevantN,
                                             stakeholder_onehot..., scenario_onehot... ]
    """
    with open(frozen_path, 'r', encoding='utf-8') as f:
        X_raw = json.load(f)
    with open(labeled_path, 'r', encoding='utf-8') as f:
        y_raw = json.load(f)
    y_map = {e['id']: e for e in y_raw}

    # --- Build deterministic canonical feature key list by scanning all alternatives ---
    feature_keys_set = set()
    for sc in X_raw:
        for alt in sc.get('alternatives', []):
            for k in alt.keys():
                if k in ('id_prod', 'c', 'wb'):
                    continue
                feature_keys_set.add(k)
    feature_keys = sorted(feature_keys_set)  # deterministic order

    # Prepare containers for per-alternative raw values and flags
    all_values = []          # will store only the 'value' entries (one per indicator)
    all_present_flags = []   # 1 if present else 0 (one per indicator)
    all_relevant_flags = []  # 1 if relevant for scenario else 0 (one per indicator)
    all_stakeholder_vecs = []
    all_scenario_vecs = []

    # First pass: extract raw triplets and split into three arrays
    for sc in X_raw:
        stakeholder_vec = encode_stakeholder_pref(sc.get('stakeholder_preference', []))
        scenario_vec = encode_scenario_pref(sc.get('scenario_preference', []))
        situations = sc.get('situations', []) or []

        # Determine relevant indicators for this scenario
        relevant_indicators = set()
        for sit in situations:
            relevant_indicators.update(PERF_MAP.get(sit, []))

        for alt in sc.get('alternatives', []):
            feats = _extract_features_from_alt(alt, feature_keys, relevant_indicators)
            # feats is [v0,p0,r0, v1,p1,r1, ..., vM,pM,rM] where M = len(feature_keys) + cost
            values = feats[0::3]
            present_flags = feats[1::3]
            relevant_flags = feats[2::3]

            all_values.append(values)
            all_present_flags.append(present_flags)
            all_relevant_flags.append(relevant_flags)
            all_stakeholder_vecs.append(stakeholder_vec)
            all_scenario_vecs.append(scenario_vec)

    # Convert to numpy arrays
    all_values = np.array(all_values, dtype=np.float32)                # (samples, N_values)
    all_present_flags = np.array(all_present_flags, dtype=np.int8)     # (samples, N_values)
    all_relevant_flags = np.array(all_relevant_flags, dtype=np.int8)   # (samples, N_values)
    all_stakeholder_vecs = np.array(all_stakeholder_vecs, dtype=np.float32)
    all_scenario_vecs = np.array(all_scenario_vecs, dtype=np.float32)

    # Scale numeric value columns using only entries that are BOTH present AND relevant
    n_cols = all_values.shape[1]
    scaled_values = np.zeros_like(all_values)

    for i in range(n_cols):
        col = all_values[:, i]
        present_col = all_present_flags[:, i]
        relevant_col = all_relevant_flags[:, i]

        #  only consider values that are BOTH present AND relevant
        valid_mask = (present_col == 1) & (relevant_col == 1)
        valid_vals = col[valid_mask]

        if valid_vals.size == 0:
            scaled_values[:, i] = 0.0
            continue

        col_min = float(valid_vals.min())
        col_max = float(valid_vals.max())

        if col_max > col_min:
            scaled_col = np.zeros_like(col)
            scaled_col[valid_mask] = (valid_vals - col_min) / (col_max - col_min)
            scaled_values[:, i] = scaled_col
        else:
            scaled_values[:, i] = 0.0

    # Rebuild per-scenario arrays, append stakeholder+scenario one-hot vectors, and attach prefs + confs
    scenarios = []
    idx = 0  # pointer into flattened alternative arrays

    # For downstream reference, provide explicit final feature_keys including 'total_cost'
    final_feature_keys = feature_keys + ['total_cost']

    for sc in X_raw:
        sid = sc.get('id')
        # keep alignment: if scenario not labeled, still advance idx and skip adding it
        if sid not in y_map:
            idx += len(sc.get('alternatives', []))
            continue

        labelled_alts = y_map[sid].get('labelled_alternatives', [])
        labelled_map = {alt['id_prod']: alt for alt in labelled_alts}

        feats_scaled = []
        prefs = []
        confs = []  

        for alt in sc.get('alternatives', []):
            scaled_vals = scaled_values[idx]            # numeric scaled values
            present_flags = all_present_flags[idx]      # 0/1
            relevant_flags = all_relevant_flags[idx]    # 0/1
            stakeholder_vec = all_stakeholder_vecs[idx]
            scenario_vec = all_scenario_vecs[idx]
            idx += 1

            combined = []
            # interleave scaled value, present flag, relevant flag per indicator
            for v, p, r in zip(scaled_vals, present_flags, relevant_flags):
                combined.append(float(v))
                combined.append(int(p))
                combined.append(int(r))

            # append stakeholder and scenario one-hot vectors (no flags)
            combined.extend(stakeholder_vec.tolist())
            combined.extend(scenario_vec.tolist())

            feats_scaled.append(combined)

            pid = alt.get('id_prod')
            if pid in labelled_map:
                labelled_entry = labelled_map[pid]
                prefs.append(labelled_entry.get('pref'))
                confs.append(labelled_entry.get('conf', None))
            else:
                prefs.append(None)
                confs.append(None)

        scenarios.append({
            'id': sid,
            'features': np.array(feats_scaled, dtype=np.float32),
            'prefs': prefs,
            'confs': confs,                   
            'feature_keys': final_feature_keys
        })

    return scenarios
