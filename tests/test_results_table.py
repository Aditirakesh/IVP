import csv

from src.eval.results_table import measures, readCsv, table

COLS = ["experiment", "protocol", "level", "metric", "mean", "sd"]


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)


def status(ms):
    return {m["name"]: m["status"] for m in ms}


def test_nothing_measured_is_pending():
    ms = measures([], [])
    assert len(ms) == 5
    assert all(m["status"] == "pending" for m in ms)
    assert "not yet measured" in table(ms)


def test_statuses(tmp_path):
    p = tmp_path / "results.csv"
    write(p, [
        ["cdl_mddnet", "kfold10", "participant", "accuracy", 0.90, 0.03],
        ["cdl_mddnet", "kfold10", "participant", "auc", 0.93, 0.02],
        ["cdl_mddnet", "loso", "participant", "accuracy", 0.70, 0.45],  # under 80
        ["cdl_mddnet_ch32", "kfold10", "participant", "accuracy", 0.88, 0.03],  # -2 points
        ["cdl_mddnet_ch16", "kfold10", "participant", "accuracy", 0.80, 0.04],
        ["cdl_mddnet_ch64", "kfold10", "participant", "accuracy", 0.91, 0.03],  # too many channels, ignored
    ])
    e = tmp_path / "eff.csv"
    e.write_text("model,channels,params,under_100k,cpu_ms,gpu_ms,cpu_threads\n"
                 "src.models.cdl_mddnet:CDLMDDNet,128,51250,True,1.3,,2\n")
    ms = measures(readCsv(str(p)), readCsv(str(e)))
    st = status(ms)
    assert st["MODMA 10-fold, cross-subject"] == "met"
    assert st["MODMA LOSO"] == "not met"
    assert st["AUC (subject-independent)"] == "met"
    assert st["Parameters"] == "met"
    assert st["Channels"] == "met"
    assert "32 ch" in [m for m in ms if m["name"] == "Channels"][0]["got"]


def test_channel_drop_too_big(tmp_path):
    p = tmp_path / "results.csv"
    write(p, [
        ["cdl_mddnet", "kfold10", "participant", "accuracy", 0.90, 0.03],
        ["cdl_mddnet_ch32", "kfold10", "participant", "accuracy", 0.85, 0.03],  # -5 points
    ])
    ms = measures(readCsv(str(p)), [])
    assert status(ms)["Channels"] == "not met"
