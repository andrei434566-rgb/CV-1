"""Serial background I/O: never block the editor or reorder review revocations."""
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Qt

_pool = None
_jobs = {}


class _Signals(QObject):
    done = Signal(object)


class _Job(QRunnable):
    def __init__(self, operation, payload):
        super().__init__()
        self.operation, self.payload = operation, payload
        self.cancelled = False
        self.signals = _Signals()

    def run(self):
        try:
            if self.cancelled:
                self.signals.done.emit({'ok':True,'operation':self.operation,'cancelled':True,'record_id':self.payload.get('record_id')})
                return
            from app.infrastructure import attribute_examples as library
            if self.operation == "save":
                result = library.write_example(**self.payload)
            elif self.operation == "similar":
                import hashlib
                from app.infrastructure.description_review import facies_candidates
                payload=dict(self.payload)
                attributes=payload.pop('attributes');assumptions=payload.pop('assumptions')
                signature=payload.pop('query_signature')
                rows=library.read_examples(root=payload.get('root'))
                image_sha=hashlib.sha256(payload['image_png']).hexdigest()
                result={'similar':library.find_similar(**payload),
                        'candidates':facies_candidates(attributes,[r for r in rows if r['id']!=payload.get('exclude_id') and r.get('sha256')!=image_sha],assumptions=assumptions),
                        'reviewed':sum(r.get('description_reviewed') is True for r in rows),
                        'query_signature':signature}
            elif self.operation == "export":
                result = library.export_attribute_dataset(**self.payload)
            elif self.operation == "revoke":
                result = library.revoke_example(**self.payload)
            elif self.operation == "quality":
                from app.infrastructure.photo_quality import audit_file
                payload=dict(self.payload);record_id=payload.pop('record_id')
                result={'record_id':record_id,'path':str(payload['path']),'quality':audit_file(**payload),
                        'regions':(payload.get('columns', []), payload.get('manual_exclusions', []))}
            elif self.operation == 'color':
                from io import BytesIO
                from PIL import Image
                from app.infrastructure.core_color_hint import estimate_core_color
                from app.infrastructure.photo_quality import label_boxes
                with Image.open(BytesIO(self.payload['image_png'])) as image:
                    import numpy as np
                    valid = np.asarray(image.getchannel('A')) > 0 if 'A' in image.getbands() else None
                    boxes=label_boxes(image,valid_mask=valid)
                    measurement=estimate_core_color(image,valid,exclude_boxes=boxes)
                result={'measurement':measurement.to_dict() if measurement else None,
                        'signature':self.payload['signature']}
            else:
                raise ValueError("Неизвестная операция базы признаков")
            reply = {"ok": True, "operation": self.operation, "result": result}
        except Exception as exc:
            reply = {"ok": False, "operation": self.operation, "error": str(exc)}
            if self.operation=='quality':reply['record_id']=self.payload.get('record_id')
        self.signals.done.emit(reply)


def submit(operation, payload, callback):
    global _pool
    if _pool is None:
        _pool = QThreadPool()
        _pool.setMaxThreadCount(1)
    job = _Job(operation, payload)
    key = id(job)
    _jobs[key] = job
    job.signals.done.connect(callback, Qt.ConnectionType.QueuedConnection)
    job.signals.done.connect(lambda _reply: _jobs.pop(key, None), Qt.ConnectionType.QueuedConnection)
    _pool.start(job)


def wait_for_tasks():
    """For shutdown/tests; normal GUI operations use asynchronous submit()."""
    if _pool is not None:
        _pool.waitForDone()


def cancel_quality_jobs(record_ids):
    """Skip queued photo reads on close; committed review writes remain queued."""
    ids=set(record_ids)
    for job in list(_jobs.values()):
        if job.operation=='quality' and job.payload.get('record_id') in ids:
            job.cancelled=True
