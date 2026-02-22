import argparse
import os
import sys
from pathlib import Path
import difflib

# Prefer local repo package over any globally installed `noise2sim`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noise2sim.tools.train import main
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    "--config-file",
    default="./configs/bsd400_unet2_ps3_ns8_gpu1.py",
    metavar="FILE",
    help="path to config file",
    type=str,
)


def _resolve_config_path(config_file):
    cfg = Path(config_file)
    if cfg.exists():
        return str(cfg)

    # Try resolving relative to repo root (scripts/..)
    repo_root = Path(__file__).resolve().parent.parent
    cfg_from_root = (repo_root / config_file).resolve()
    if cfg_from_root.exists():
        return str(cfg_from_root)

    # Backward compatibility: fallback from old DM4 config name to new alias
    if cfg.name == 'pcct_dm4_unet2.py':
        fallback = repo_root / 'configs' / 'pcct_tem_dm4_unet2.py'
        if fallback.exists():
            print("[Info] '{}' not found. Falling back to '{}'".format(config_file, fallback))
            return str(fallback)

    # Helpful error with available configs and nearest match
    config_dir = repo_root / 'configs'
    available = sorted([p.name for p in config_dir.glob('*.py')]) if config_dir.exists() else []
    closest = difflib.get_close_matches(cfg.name, available, n=3)
    hint = ''
    if closest:
        hint = '\nDid you mean one of these? {}'.format(', '.join(closest))
    if available:
        hint += '\nAvailable config files under ./configs/:\n- ' + '\n- '.join(available)

    raise FileNotFoundError(
        "Config file '{}' does not exist.\n"
        "Current working directory: {}\n"
        "Try: python ./scripts/train.py --config-file ./configs/pcct_dm4_unet2.py{}".format(
            config_file, os.getcwd(), hint
        )
    )




def _debug_print_import_path():
    if os.environ.get("NOISE2SIM_DEBUG_IMPORT", "0") == "1":
        import noise2sim
        print("[Debug] using noise2sim package at:", Path(noise2sim.__file__).resolve())

def train():
    args = parser.parse_args()
    _debug_print_import_path()
    config_file = _resolve_config_path(args.config_file)
    main(config_file)


if __name__ == '__main__':
    train()
