"""Run final-paper experiments into a separate, ignored output directory."""
from pathlib import Path
import argparse
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--with-e5', action='store_true')
    args = parser.parse_args()
    required = ['uber_reviews_without_reviewid.csv', 'data/processed/google_play_app_reviews_standardized.csv']
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        parser.error('Missing inputs: ' + ', '.join(missing))
    if args.check_inputs:
        print('Required local inputs are present. Experiments were not run.')
        return
    refresh = load('final_protocol_refresh', 'train_only_protocol_refresh.py')
    refresh.OUT = ROOT / 'reproduced' / 'reports'
    refresh.OUT.mkdir(parents=True, exist_ok=True)
    if args.with_e5:
        original_e5 = refresh.run_e5_corrected
        def e5_with_cache():
            embedding = refresh.load_script('cache_encoder', '34_bert_card_eval.py')
            slug = embedding.MODEL_NAME.replace('/', '__').replace('-', '_')
            for name, relative in [('uber', required[0]), ('google_play_app_reviews', required[1])]:
                cache = refresh.PROCESSED / f'embedding_card_{slug}_{name}_canonical_20pct.npy'
                if not cache.exists():
                    raw = refresh.pd.read_csv(ROOT / relative)
                    real = refresh.el.prepare_real_reviews(raw)
                    split = refresh.el.split_real_reviews(real, seed=42)
                    attacks = refresh.el.generate_attacks(real, int(round(len(split.train_real) * .20)), seed=4400)
                    full = refresh.pd.concat([real, attacks], ignore_index=True)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    embedding.encode_texts(full['content'].astype(str).tolist(), cache)
            original_e5()
        refresh.run_e5_corrected = e5_with_cache
        refresh.main()
    else:
        for filename, name, canonical, explicit in [
            ('07_attack_sweep.py', 'sweep', True, False),
            ('31_trusted_calibrated_card_eval.py', 'calibration', True, False),
            ('20_seed_stability.py', 'seeds', False, True),
            ('42_strong_noisy_label_baselines.py', 'baselines', True, False),
            ('33_cross_dataset_card_eval.py', 'cross', True, False),
            ('39_scale_stress_eval.py', 'scale', False, True),
            ('40_reliability_interaction_diagnostics.py', 'formula', True, False),
            ('45_cohort_weight_simplex_search.py', 'oracle', True, False),
            ('19_cohort_feature_ablation.py', 'features', True, False),
            ('18_hyperparameter_sensitivity.py', 'parameters', True, False),
        ]:
            refresh.run_patched(filename, name, canonical=canonical, explicit=explicit)
        refresh.run_review_examples()
        refresh.run_attack_types()
    for name, filename in [('natural', '46_natural_burst_false_positive.py'),
                           ('adaptive', '47_partially_adaptive_mixed_eval.py')]:
        module = load(name, filename)
        module.RESULTS = ROOT / 'reproduced' / 'results'
        module.main()
    print('Finished. Outputs are under reproduced/. Recorded paper results were preserved.')

if __name__ == '__main__':
    main()
