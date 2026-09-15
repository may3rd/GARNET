import unittest

try:
    import api
except ModuleNotFoundError as exc:
    if exc.name == "pdf2image":
        api = None
    else:
        raise

from garnet.pid_extractor import PIDPipeline


@unittest.skipIf(api is None, "pdf2image is not installed in this test environment")
class PipelineStageOrderTests(unittest.TestCase):
    def test_stage_order_matches_pid_extractor_definitions(self) -> None:
        # _stage_definitions() only builds a list of (num, name, bound_method)
        # tuples from `self.<method>` attribute lookups -- it never calls the
        # methods or reads any instance state, so a bare __new__() instance
        # (skipping __init__) is safe to use here.
        pipeline = PIDPipeline.__new__(PIDPipeline)
        expected = [(num, name) for num, name, _fn in pipeline._stage_definitions()]
        self.assertEqual(api.PIPELINE_STAGE_ORDER, expected)

if __name__ == "__main__":
    unittest.main()
