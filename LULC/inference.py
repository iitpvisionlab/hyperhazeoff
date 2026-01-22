"""
HyperHazeOff LULC Inference Tool.
Simple, clean version. Reads YAML config.
"""

import argparse
import logging
import os
import sys
import time
import json
from pathlib import Path
from typing import Any, Optional

import yaml

# --- Project Imports ---
try:
    from LULC.data import FineTuningDataGenerator
    from LULC.evaluate import eval_model_by_class, aggregate_predictions, dump_class_report_json
    import tensorflow as tf
except ImportError:
    PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[1]
    sys.path.append(str(PROJECT_ROOT_FALLBACK))
    from LULC.data import FineTuningDataGenerator
    from LULC.evaluate import eval_model_by_class, aggregate_predictions, dump_class_report_json
    import tensorflow as tf

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Inference")

def setup_device(device: str) -> None:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception: pass
        logger.info("Compute Mode: Force CPU")
    else:
        logger.info("Compute Mode: GPU")

def load_yaml_config(path: Path) -> dict:
    """Loads YAML configuration."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_path(cli_arg: Optional[Path], config_val: Optional[str], name: str) -> Path:
    """Simple resolver: CLI > Config > Error."""
    if cli_arg is not None:
        return cli_arg
    if config_val:
        return Path(config_val)
    raise ValueError(f"Path for '{name}' not specified.")

def main() -> None:
    # Default config location
    project_root = Path(__file__).resolve().parents[1]
    default_config = project_root / "meta" / "LULC" / "config.yaml"  # <--- .yaml

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--config", type=Path, default=default_config, help="Path to config.yaml")
    
    # Overrides
    parser.add_argument("--input-dir", type=Path, help="Override input dir")
    parser.add_argument("--annotations-dir", type=Path, help="Override annotations dir")
    parser.add_argument("--model-path", type=Path, help="Override model path")
    parser.add_argument("--output-dir", type=Path, help="Override output dir")
    
    # Params
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--save-preds", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Clean mode (no haze)")

    args = parser.parse_args()
    setup_device(args.device)

    # 1. Load Config (YAML)
    logger.info(f"Loading config: {args.config}")
    try:
        config = load_yaml_config(args.config)
        paths = config.get("paths", {})
        classes = config.get("classes")
        if not classes: raise ValueError("Missing 'classes' list in config")
    except Exception as e:
        logger.critical(f"Config error: {e}")
        sys.exit(1)

    # 2. Resolve Paths
    try:
        input_dir = get_path(args.input_dir, paths.get("input_dir"), "Input Dir")
        ann_dir = get_path(args.annotations_dir, paths.get("annotations_dir"), "Annotations Dir")
        model_path = get_path(args.model_path, paths.get("model_path"), "Model Path")
        
        out_val = args.output_dir if args.output_dir else paths.get("output_dir", "results")
        output_dir = Path(out_val)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # 3. Prepare
    output_dir.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        logger.critical(f"Model not found: {model_path}")
        sys.exit(1)

    # 4. Run
    model_params = config.get("model_params", {})
    num_channels = model_params.get("input_channels", 9)
    
    logger.info(f"Input: {input_dir}")
    
    data_gen = FineTuningDataGenerator(
        images_dir=input_dir,
        annotations_dir=ann_dir,
        num_channels=num_channels,
        clean=args.clean,
        batch_size=args.batch_size
    )

    if len(data_gen) == 0:
        logger.warning("Dataset empty.")
        sys.exit(0)

    logger.info(f"Starting inference (Model: {model_path.name})...")
    start = time.time()

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        metrics = eval_model_by_class(model, data_gen, classes)
        
        if args.save_preds:
            aggregate_predictions(model, data_gen, classes, str(output_dir / "preds.json"))
            
    except Exception as e:
        logger.critical(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)

    # 5. Report
    dump_class_report_json(
        metrics, 
        str(output_dir / "report.json"),
        run_info={"config": str(args.config), "clean": args.clean, "device": args.device}
    )

    print("\n" + "="*40 + "\n RESULTS \n" + "="*40)
    if hasattr(metrics, "to_string"):
        print(metrics.to_string())
    else:
        print(json.dumps(metrics, indent=2, default=str))
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
