"""CLI entry point for Real-Time License Plate Recognition."""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import ALPRPipeline


def main():
    parser = argparse.ArgumentParser(description="Real-Time ALPR System")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--source", default=None, help="Video source (overrides config)")
    args = parser.parse_args()

    pipeline = ALPRPipeline(config_path=args.config)

    if args.source:
        pipeline.config["video"]["source"] = int(args.source) if args.source.isdigit() else args.source

    print("Starting ALPR Pipeline...")
    print("Press 'q' to quit")
    pipeline.run()


if __name__ == "__main__":
    main()
