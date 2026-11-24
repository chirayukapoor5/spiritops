from spiritops.demonstration import demonstrate


def test_real_http_model_lifecycle(tmp_path):
    report = demonstrate(tmp_path, "fixture")
    assert report["bootstrap"]["outcome"] == "PROMOTE"
    assert report["monitoring"][-1]["action"] == "RETRAIN"
    assert report["performance_before_retraining"]["metrics"]["samples"] > 0
    assert report["automatic_job"]["outcome"] in {"PROMOTE", "REJECT_CANDIDATE"}
    assert report["baseline_challenger"]["outcome"] == "REJECT_CANDIDATE"
    assert report["requests_during_training_and_activation"]["errors"] == 0
    assert report["concurrent"]["errors"] == 0
    assert report["after_restart"]["prediction"] >= 0
    if "rollback" in report:
        assert report["rollback"]["from"] != report["rollback"]["restored_version"]
        assert report["after_restart"]["version"] == report["rollback"]["restored_version"]
