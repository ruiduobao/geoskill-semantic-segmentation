"""CLI argument parsing tests for semantic-segmentation."""
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from conftest import mod, SCRIPT


def run_cli(args, timeout=90):
    return subprocess.run(
        [sys.executable, SCRIPT] + args,
        capture_output=True, text=True, timeout=timeout,
    )


class TestCLIBasics:
    def test_help(self):
        r = run_cli(["--help"])
        assert r.returncode == 0
        assert "--bbox" in r.stdout
        assert "--synthetic" in r.stdout
        assert "--n-classes" in r.stdout

    def test_version(self):
        r = run_cli(["--version"])
        assert r.returncode == 0
        assert "1.0.0" in r.stdout

    def test_no_args_exit_2(self):
        r = run_cli([])
        assert r.returncode == 2

    def test_bad_method_exit_2(self):
        r = run_cli(["--bbox", "116", "39", "117", "40", "--synthetic", "--method", "bad"])
        assert r.returncode == 2

    def test_input_file_not_found_exit_2(self):
        r = run_cli(["--input", "nonexistent.tif"])
        assert r.returncode == 2

    def test_synthetic_without_bbox_exit_2(self):
        r = run_cli(["--synthetic"])
        assert r.returncode == 2

    def test_synthetic_rf_uses_truth_labels(self, tmp_path):
        """合成模式 + rf：用内置真值做有监督训练，应成功。"""
        out = str(tmp_path / "out_rf")
        r = run_cli(["--bbox", "116", "39", "117", "40", "--synthetic",
                     "--method", "rf", "--output-dir", out, "--quiet"])
        assert r.returncode == 0, r.stderr

    def test_rf_without_labels_real_mode_exit_2(self, tmp_path):
        """真实模式 + rf 但无 --labels -> UsageError exit 2。"""
        import numpy as np
        scene = str(tmp_path / "scene.tif")
        mod.write_geotiff(scene, np.random.uniform(0, 1, (4, 16, 16)).astype(np.float32),
                          [116, 39, 117, 40])
        r = run_cli(["--input", scene, "--method", "rf"])
        assert r.returncode == 2
