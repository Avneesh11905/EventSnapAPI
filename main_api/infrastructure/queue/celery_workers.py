from celery import shared_task
import asyncio
from infrastructure.di_container import get_container

@shared_task(bind=True, name="encode_event_task", acks_late=True)
def encode_event_task(self, folder_path: str, max_faces: int = 0, det_conf: float = 0.5, nms_thresh: float = 0.4):
    container = get_container()
    use_case = container.process_event_encoding_use_case()
    
    def update_state_cb(state_name, meta_dict):
        self.update_state(state=state_name, meta=meta_dict)
        
    return asyncio.run(use_case.execute(
        folder_path=folder_path,
        max_faces=max_faces,
        det_conf=det_conf,
        nms_thresh=nms_thresh,
        update_state_cb=update_state_cb
    ))

@shared_task(bind=True, name="create_event_zip_task", acks_late=True)
def create_event_zip_task(self, event_id: str, user_id: str, image_paths: list[dict]):
    container = get_container()
    use_case = container.create_event_zip_use_case()
    
    def update_state_cb(state_name, meta_dict):
        self.update_state(state=state_name, meta=meta_dict)
        
    return asyncio.run(use_case.execute(
        event_id=event_id,
        user_id=user_id,
        image_paths=image_paths,
        update_state_cb=update_state_cb
    ))
