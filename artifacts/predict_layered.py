import joblib
import pandas as pd
import numpy as np
import features

# Load models
l1_models = joblib.load("l1_models.joblib")
l2_model = joblib.load("l2_models.joblib")

def predict_layered(response_dict, meeting_dict, top_k=5):
    # 1. Extract candidates & compute features
    cands = features.build_candidates(response_dict, meeting_dict)
    if cands.empty:
        return []
    feats = features.build_features(cands, response_dict, meeting_dict)
    
    # 2. Predict base L1 scores
    l1_preds = {}
    feature_names = list(feats.columns)
    for m_name, model in l1_models.items():
        if m_name in ['gb', 'lgb']:
            l1_preds[m_name] = model.predict(feats)
        else:
            l1_preds[m_name] = model.predict_proba(feats)[:, 1]
            
    # 3. Form L2 features
    meta_df = pd.DataFrame(l1_preds)
    pruned_meta = ['gb', 'lgb', 'rf']
    for m in pruned_meta:
        meta_df[f"{m}_z"] = (meta_df[m] - meta_df[m].mean()) / (meta_df[m].std() + 1e-9)
        meta_df[f"{m}_pct"] = meta_df[m].rank(pct=True)
        
    # 4. Injected rule baseline
    meta_df['pred_rule'] = (
        2 * feats['pref_day'] - 2 * feats['avoid_day'] + 2 * feats['pref_time_share']
        - 2 * feats['avoid_time_share'] - 1.5 * feats['break_shortfall']
        - 0.5 * feats['b2b_x_tolerance'] * feats['back_to_back'] + 1.0 * feats['modality_match']
        - feats['extra_trip_cost']/60.0 - 0.5 * feats['overlaps_lunch'] - 0.5 * feats['is_saturday']
    )
    
    # 5. Layer 2 Scoring & Layer 3 Soft Penalty Injection
    meta_cols = [f"{m}_z" for m in pruned_meta] + [f"{m}_pct" for m in pruned_meta] + ['pred_rule']
    p_lgb_meta = l2_model.predict(meta_df[meta_cols])
    
    # Soft blend
    p_l3 = 0.6 * p_lgb_meta + 0.4 * meta_df['pred_rule']
    
    # Soft avoidance penalty deduction
    avoid_mask = (feats['avoid_day'] == 1.0) | (feats['avoid_time_share'] > 0.5)
    p_l3[avoid_mask] -= 0.5
    
    cands['score'] = p_l3
    return cands.sort_values(by='score', ascending=False).head(top_k).to_dict(orient="records")
