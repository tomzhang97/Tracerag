import os
import yaml
from pathlib import Path

def setup_configs():
    base_dir = Path("configs")
    base_dir.mkdir(exist_ok=True)
    
    # Models
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    with open(models_dir / "tracerag_full.yaml", "w") as f:
        yaml.dump({
            "name": "tracerag_full",
            "retrieval": {"use_snapper": True},
            "visual": {"enabled": True, "model_name": "vidore/colpali-v1.2"},
            "structural": {"enabled": True},
            "revision": {"enabled": True},
            "llm": {"enabled": True}
        }, f, sort_keys=False)

    with open(models_dir / "tracerag_core.yaml", "w") as f:
        yaml.dump({
            "name": "tracerag_core",
            "retrieval": {"use_snapper": True},
            "visual": {"enabled": True, "model_name": "vidore/colpali-v1.2"},
            "structural": {"enabled": True},
            "revision": {"enabled": True},
            "llm": {"enabled": False}
        }, f, sort_keys=False)
        
    with open(models_dir / "tracerag_qa.yaml", "w") as f:
        yaml.dump({
            "name": "tracerag_qa",
            "extends": "tracerag_full"
        }, f, sort_keys=False)

    # Baselines
    baselines_dir = base_dir / "baselines"
    baselines_dir.mkdir(exist_ok=True)
    
    for b in ["tesseract", "paddleocr", "colpali", "colqwen2", "gpt4o", "absdiff", "siamese_diff"]:
        with open(baselines_dir / f"{b}.yaml", "w") as f:
            yaml.dump({"name": b, "type": "baseline"}, f)

    # Ablations
    ablations_dir = base_dir / "ablations"
    ablations_dir.mkdir(exist_ok=True)
    
    ablations = [
        ("no_snap", ["no_snap"]),
        ("no_vector", ["no_vector"]),
        ("no_revision", ["no_revision"]),
        ("no_vision", ["no_vision"]),
        ("localize_only", ["no_vision", "no_snap"])
    ]
    
    for name, overrides in ablations:
        with open(ablations_dir / f"{name}.yaml", "w") as f:
            yaml.dump({
                "name": f"tracerag_{name}",
                "base_model": "tracerag_full",
                "ablations": overrides
            }, f, sort_keys=False)
            
    print("Configs generated!")

if __name__ == "__main__":
    setup_configs()
