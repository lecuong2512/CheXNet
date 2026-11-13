"""Observe real PyTorch forward calls without changing model inputs or outputs."""
from datetime import datetime, timezone
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import threading
import time
from uuid import uuid4

logger = logging.getLogger('chexnet.inference')
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('[INFER] %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def configure_audit_log(path):
    destination = Path(path).resolve()
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler) and handler.baseFilename == str(destination):
            return handler
    destination.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(destination, maxBytes=5_000_000, backupCount=3, encoding='utf-8')
    handler.setFormatter(logging.Formatter('[INFER] %(message)s'))
    logger.addHandler(handler)
    return handler


def tensor_shape(value):
    if hasattr(value, 'shape'):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        for item in value:
            shape = tensor_shape(item)
            if shape is not None:
                return shape
    return None


class ModelCallTrace:
    def __init__(self, hybrid_model, yolo_model=None):
        self.id = 'inf_' + uuid4().hex
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started = time.perf_counter()
        self.owner = threading.get_ident()
        self.handles = []
        self.models = {}
        self.targets = {'hybrid': hybrid_model, 'yolo': yolo_model}
        self.result = None

    def __enter__(self):
        logger.info('id=%s event=request_started', self.id)
        for name, model in self.targets.items():
            stage = {'modelClass': type(model).__name__ if model is not None else None,
                     'forwardCalls': 0, 'observed': False, 'durationMs': 0.0}
            self.models[name] = stage
            before = getattr(model, 'register_forward_pre_hook', None)
            after = getattr(model, 'register_forward_hook', None)
            if not callable(before) or not callable(after):
                continue
            pending = []

            def pre_hook(module, inputs, *, key=name, times=pending):
                if threading.get_ident() != self.owner:
                    return
                times.append(time.perf_counter())
                self.models[key]['inputShape'] = tensor_shape(inputs)
                logger.info('id=%s model=%s event=forward_started input=%s', self.id, key, self.models[key]['inputShape'])

            def post_hook(module, inputs, output, *, key=name, times=pending):
                if threading.get_ident() != self.owner:
                    return
                record = self.models[key]
                record['durationMs'] += (time.perf_counter() - times.pop()) * 1000 if times else 0
                record['forwardCalls'] += 1
                record['observed'] = True
                record['outputShape'] = tensor_shape(output)
                logger.info('id=%s model=%s event=forward_completed output=%s calls=%s',
                            self.id, key, record['outputShape'], record['forwardCalls'])

            registered = []
            try:
                registered.append(before(pre_hook))
                registered.append(after(post_hook))
            except Exception:
                for handle in registered:
                    handle.remove()
                logger.warning('id=%s model=%s event=observer_unavailable', self.id, name)
            else:
                self.handles.extend(registered)
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        for stage in self.models.values():
            stage['durationMs'] = round(stage['durationMs'], 3)
        status = 'completed' if exc_type is None else 'failed'
        self.result = {'id': self.id, 'startedAt': self.started_at,
                       'completedAt': datetime.now(timezone.utc).isoformat(),
                       'durationMs': round((time.perf_counter() - self.started) * 1000, 3),
                       'status': status, 'models': self.models}
        logger.info('id=%s event=request_%s hybrid_calls=%s yolo_calls=%s duration_ms=%s',
                    self.id, status, self.models['hybrid']['forwardCalls'],
                    self.models['yolo']['forwardCalls'], self.result['durationMs'])
        return False
