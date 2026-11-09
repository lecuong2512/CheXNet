from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

import torch

from Backend.ai_service.inference_trace import ModelCallTrace, configure_audit_log, logger


class InferenceTraceTests(unittest.TestCase):
    def test_audit_log_is_appended_and_not_duplicated_when_configured_again(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'audit.log'
            path.write_text('previous entry\n', encoding='utf-8')
            handler = configure_audit_log(path)
            try:
                self.assertIs(handler, configure_audit_log(path))
                logger.info('audit-test-entry')
                text = path.read_text(encoding='utf-8')
                self.assertIn('previous entry', text)
                self.assertEqual(text.count('audit-test-entry'), 1)
            finally:
                logger.removeHandler(handler)
                handler.close()

    def test_real_forward_is_observed_without_changing_output_or_leaking_hooks(self):
        hybrid = torch.nn.Linear(3, 15)
        yolo = torch.nn.Linear(3, 4)
        inputs = torch.ones(1, 3)
        expected = hybrid(inputs)
        with self.assertLogs('chexnet.inference', level='INFO') as logs:
            with ModelCallTrace(hybrid, yolo) as trace:
                actual = hybrid(inputs)
                yolo(inputs)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(trace.result['status'], 'completed')
        self.assertTrue(trace.result['models']['hybrid']['observed'])
        self.assertEqual(trace.result['models']['hybrid']['forwardCalls'], 1)
        self.assertEqual(trace.result['models']['hybrid']['inputShape'], [1, 3])
        self.assertEqual(trace.result['models']['hybrid']['outputShape'], [1, 15])
        self.assertEqual(trace.result['models']['yolo']['forwardCalls'], 1)
        self.assertTrue(all(trace.id in message for message in logs.output))
        self.assertEqual(len(hybrid._forward_hooks), 0)
        self.assertEqual(len(hybrid._forward_pre_hooks), 0)
        hybrid(inputs)
        self.assertEqual(trace.result['models']['hybrid']['forwardCalls'], 1)

    def test_failed_forward_is_not_claimed_as_completed_and_hooks_are_removed(self):
        class Failing(torch.nn.Module):
            def forward(self, inputs):
                raise RuntimeError('controlled failure')
        model = Failing()
        trace = ModelCallTrace(model)
        with self.assertRaisesRegex(RuntimeError, 'controlled failure'), self.assertLogs('chexnet.inference', level='INFO'):
            with trace:
                model(torch.ones(1, 3))
        self.assertEqual(trace.result['status'], 'failed')
        self.assertFalse(trace.result['models']['hybrid']['observed'])
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertEqual(len(model._forward_hooks), 0)

    def test_calls_on_other_threads_do_not_contaminate_the_request(self):
        model = torch.nn.Linear(3, 15)
        with self.assertLogs('chexnet.inference', level='INFO'):
            with ModelCallTrace(model) as trace:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(model, torch.ones(1, 3)).result(timeout=5)
                model(torch.ones(1, 3))
        self.assertEqual(trace.result['models']['hybrid']['forwardCalls'], 1)

    def test_no_observer_does_not_make_a_model_execution_claim(self):
        with self.assertLogs('chexnet.inference', level='INFO'):
            with ModelCallTrace(object()) as trace:
                pass
        self.assertFalse(trace.result['models']['hybrid']['observed'])
        self.assertFalse(trace.result['models']['yolo']['observed'])

    def test_observer_registration_failure_keeps_inference_available_and_cleans_up(self):
        class Unobserved(torch.nn.Linear):
            def register_forward_hook(self, hook):
                raise RuntimeError('hook unavailable')
        model = Unobserved(3, 15)
        with self.assertLogs('chexnet.inference', level='INFO'):
            with ModelCallTrace(model) as trace:
                output = model(torch.ones(1, 3))
        self.assertEqual(list(output.shape), [1, 15])
        self.assertFalse(trace.result['models']['hybrid']['observed'])
        self.assertEqual(len(model._forward_pre_hooks), 0)


if __name__ == '__main__':
    unittest.main()
