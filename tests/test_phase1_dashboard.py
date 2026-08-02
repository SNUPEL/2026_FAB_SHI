import json
import sys
import tempfile
import unittest
from pathlib import Path

# scripts/ 를 import 경로에 올린다 (build_phase1_dashboard 와 그 의존 build_training_dashboard 가 여기 있다).
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import build_phase1_dashboard as dash  # noqa: E402
import watch_phase1_dashboard as watch  # noqa: E402


_METRICS = (
    "episode,loss,best_source\n"
    "1,2.5,NP_NC:wo_first_balanced|FN_FL:agent_greedy\n"
    "2,2.1,NP_NC:agent_sample_5|FN_FL:cut_first_balanced\n"
    "3,1.9,NP_NC:agent_greedy|FN_FL:agent_sample_9\n"
)

_VALIDATION = (
    "train_episode,validation_view,agent_is_best,agent_rank\n"
    "100,NP_NC,1,1\n"
    "100,FN_FL,0,2\n"
    "200,NP_NC,1,1\n"
    "200,FN_FL,1,1\n"
)


def _make_run(tmp: Path) -> Path:
    run = tmp / "run"
    run.mkdir()
    (run / "metrics.csv").write_text(_METRICS, encoding="utf-8")
    (run / "validation_summary.csv").write_text(_VALIDATION, encoding="utf-8")
    return run


class TestPhase1BestRateByView(unittest.TestCase):
    def test_np_nc_view_series(self):
        rows = dash._read_csv_rows_for_test(_VALIDATION)
        series = dash.phase1_best_rate_by_view(rows, "NP_NC")
        self.assertEqual(series, [[100.0, 100.0], [200.0, 100.0]])

    def test_fn_fl_view_series(self):
        rows = dash._read_csv_rows_for_test(_VALIDATION)
        series = dash.phase1_best_rate_by_view(rows, "FN_FL")
        self.assertEqual(series, [[100.0, 0.0], [200.0, 100.0]])

    def test_overall_pools_rows_not_view_average(self):
        rows = dash._read_csv_rows_for_test(_VALIDATION)
        series = dash.phase1_best_rate_by_view(rows, "overall")
        # ep100: rows NP_NC=1, FN_FL=0 -> 50%. ep200: 1,1 -> 100%.
        self.assertEqual(series, [[100.0, 50.0], [200.0, 100.0]])


class TestPhase1AdoptionSeries(unittest.TestCase):
    def test_per_episode_fractions_window_one(self):
        rows = dash._read_csv_rows_for_test(_METRICS)
        adoption = dash.phase1_adoption_series(rows, window=1)
        # ep1: greedy(FN_FL) + heuristic(NP_NC) -> greedy 50, sample 0, heur 50
        self.assertEqual(adoption["greedy"][0], [1.0, 50.0])
        self.assertEqual(adoption["sample"][0], [1.0, 0.0])
        self.assertEqual(adoption["heuristic"][0], [1.0, 50.0])
        # ep2: sample(NP_NC) + heuristic(FN_FL) -> greedy 0, sample 50, heur 50
        self.assertEqual(adoption["greedy"][1], [2.0, 0.0])
        self.assertEqual(adoption["sample"][1], [2.0, 50.0])
        # ep3: greedy + sample -> greedy 50, sample 50, heur 0
        self.assertEqual(adoption["greedy"][2], [3.0, 50.0])
        self.assertEqual(adoption["heuristic"][2], [3.0, 0.0])


class TestBuildPayload(unittest.TestCase):
    def test_payload_shape(self):
        with tempfile.TemporaryDirectory() as td:
            run = _make_run(Path(td))
            payload = dash.build_payload(run, target=20000, window=25)
        self.assertEqual(payload["window"], 25)
        run_data = payload["run"]
        self.assertEqual(run_data["current_ep"], 3)
        self.assertEqual(run_data["target"], 20000)
        for key in ("best_rate_np_nc", "best_rate_fn_fl", "best_rate_overall", "adoption", "best_rate_latest"):
            self.assertIn(key, run_data)
        self.assertEqual(run_data["best_rate_latest"], 100.0)  # overall 최신(ep200)


class TestWriteDashboard(unittest.TestCase):
    def test_writes_self_contained_html_and_json(self):
        with tempfile.TemporaryDirectory() as td:
            run = _make_run(Path(td))
            out = Path(td) / "dash" / "dashboard.html"
            payload = dash.write_dashboard(run, target=20000, output=out, window=25)
            self.assertTrue(out.exists())
            html = out.read_text(encoding="utf-8")
            self.assertNotIn("__DATA__", html)          # 데이터가 실제로 치환됐다
            self.assertIn('id="app"', html)
            self.assertIn('id="chart-val"', html)
            self.assertIn('id="chart-adopt"', html)
            # JSON 인코딩된 형태로 비교한다: Windows 경로는 백슬래시가 \\ 로 이스케이프되어
            # render_html 이 내장하는 실제 문자열은 str(run) 그대로가 아니다(POSIX 경로는 무영향).
            self.assertIn(json.dumps(str(run))[1:-1], html)  # run 경로가 푸터에 박힌다
            data_file = out.parent / "dashboard_data.json"
            self.assertTrue(data_file.exists())
            reloaded = json.loads(data_file.read_text(encoding="utf-8"))
            self.assertEqual(reloaded["run"]["current_ep"], payload["run"]["current_ep"])


class TestWatcherPolling(unittest.TestCase):
    def test_latest_validation_episode(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            (run / "validation_summary.csv").write_text(_VALIDATION, encoding="utf-8")
            self.assertEqual(watch.latest_validation_episode(run), 200)

    def test_latest_validation_episode_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(watch.latest_validation_episode(Path(td)))

    def test_current_train_episode(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            (run / "metrics.csv").write_text(_METRICS, encoding="utf-8")
            self.assertEqual(watch.current_train_episode(run), 3)

    def test_current_train_episode_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(watch.current_train_episode(Path(td)))


if __name__ == "__main__":
    unittest.main()
