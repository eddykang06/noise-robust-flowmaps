from dataclasses import replace

import pytest

from src.config import load

ARGS = ["slug=test", "objective.data_proportion=0.75", "sampler.steps=[1,4]"]


def test_overrides_are_nested_and_typed():
    cfg = load("configs/toy.yaml", ARGS)
    assert cfg.objective["data_proportion"] == 0.75
    assert cfg.sampler["steps"] == [1, 4]


def test_run_id_is_deterministic_and_depends_on_content():
    cfg = load("configs/toy.yaml", ARGS)
    assert cfg.run_id == load("configs/toy.yaml", ARGS).run_id
    assert cfg.run_id != replace(cfg, seed=43).run_id


def test_run_id_ignores_output_paths():
    cfg = load("configs/toy.yaml", ARGS)
    assert cfg.run_id == replace(cfg, out_root="/elsewhere", runs_dir="other").run_id


def test_paths_expand_environment(monkeypatch):
    monkeypatch.setenv("NRFM_OUT", "/scratch/x")
    assert load("configs/toy.yaml", ARGS).out_root == "/scratch/x"


def test_unset_path_variable_raises(monkeypatch):
    monkeypatch.delenv("NRFM_OUT")
    with pytest.raises(ValueError, match="NRFM_OUT"):
        load("configs/toy.yaml", ARGS)
