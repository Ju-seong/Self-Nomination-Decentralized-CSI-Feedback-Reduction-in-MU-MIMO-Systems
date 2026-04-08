"""
Quick check that selfnomination_project imports and main/test entry points are valid.
Run from the selfnomination_project directory: python check_imports.py
"""
from __future__ import print_function
import sys
import os

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    os.chdir(here)

    errors = []
    # 1) Config
    try:
        from config import (
            num_users, K, M, Nt, num_epochs, batch_size,
            save_model_path, save_testresult_path,
            ensure_file_directory_exists,
        )
        print("config: ok")
    except Exception as e:
        errors.append(("config", e))
        print("config: FAIL", e)

    # 2) Loaders (needs numpy; RF path needs no files)
    try:
        from loaders import CommunicationDataset
        d = CommunicationDataset("RF")
        assert d.num_samples > 0 and d.num_users == num_users
        print("loaders (RF): ok")
    except Exception as e:
        errors.append(("loaders", e))
        print("loaders: FAIL", e)

    # 3) Baseline methods
    try:
        from baseline_methods.sched_bf_modules import (
            random_scheduling, topK_scheduling,
            zf_beamforming, rzf_beamforming, sum_rate_calculation,
        )
        print("baseline_methods: ok")
    except Exception as e:
        errors.append(("baseline_methods", e))
        print("baseline_methods: FAIL", e)

    # 4) Learning modules
    try:
        from learning_modules import get_module_class, REINFORCE_FullInput
        C = get_module_class("reinforce", "full")
        assert C is REINFORCE_FullInput
        print("learning_modules: ok")
    except Exception as e:
        errors.append(("learning_modules", e))
        print("learning_modules: FAIL", e)

    # 5) main entry
    try:
        from main import get_model_class, get_model_save_path
        M = get_model_class("reinforce", "full")
        p = get_model_save_path("reinforce", "full", "greedy", "rzf", "RF")
        assert "REINFORCE" in p and "best.pth" in p
        print("main (entry): ok")
    except Exception as e:
        errors.append(("main", e))
        print("main: FAIL", e)

    # 6) test_unified entry (load_model, get_model_class, etc.)
    try:
        from test_unified import (
            get_model_class as t_get_model_class,
            load_model,
            BaselineMethod,
        )
        # Baseline path (no model file)
        m = load_model(None, "baseline", "full", "greedy", "zf", None)
        assert isinstance(m, BaselineMethod)
        print("test_unified (entry): ok")
    except Exception as e:
        errors.append(("test_unified", e))
        print("test_unified: FAIL", e)

    if errors:
        print("\nSome checks failed:", [n for n, _ in errors])
        sys.exit(1)
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
