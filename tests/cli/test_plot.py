import sys

from sae_pipeline.cli import plot
from sae_pipeline.eval.output import evaluation_directory


def test_plot_reads_same_partition_language_and_tag_as_evaluation(tmp_path, monkeypatch):
    config = tmp_path / "cfg.yaml"
    config.write_text(f"run_id: run\ndata: {{total_tokens: 10}}\ntarget: {{layer: 0, component: resid_post}}\nlog:\n  log_dir: {tmp_path / 'logs'}\n")
    directory = evaluation_directory(tmp_path / "logs", "variant", "L0_resid_post", "validation", "de", "best")
    directory.mkdir(parents=True)
    summary = directory / "jumprelu_w8_l0_2_eval.json"
    summary.write_text("{}")
    calls = []
    monkeypatch.setattr(plot, "plot_eval_summary", lambda **kw: calls.append(kw))
    monkeypatch.setattr(sys, "argv", ["plot", "--config", str(config), "--layer", "0", "--component", "resid_post", "--arch", "jumprelu", "--width", "8", "--l0", "2", "--eval-only", "--partition", "validation", "--language", "de", "--output-id", "variant", "--output-tag", "best"])
    plot.main()
    assert calls[0]["eval_json_path"] == summary
