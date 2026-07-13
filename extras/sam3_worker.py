import argparse
import base64
import contextlib
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image


def combine_masks(masks, scores, image_shape, confidence_threshold=0.3, max_detections=0):
    masks = np.asarray(masks)
    scores = np.asarray(scores).reshape(-1)

    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3 or len(masks) != len(scores):
        raise ValueError('SAM 3 returned incompatible masks and scores.')

    order = np.argsort(scores)[::-1]
    order = order[scores[order] >= confidence_threshold]
    if max_detections > 0:
        order = order[:max_detections]

    if len(order) == 0:
        union_mask = np.zeros(image_shape, dtype=np.uint8)
    else:
        union_mask = np.any(masks[order] > 0, axis=0).astype(np.uint8) * 255
    return union_mask, order


class SAM3Runtime:
    def __init__(self, checkpoint=None, keep_gpu=False):
        self.checkpoint = checkpoint
        self.keep_gpu = keep_gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.lock = threading.Lock()

    def load(self):
        if self.model is not None:
            return

        from sam3.model_builder import build_sam3_image_model

        kwargs = {'device': 'cpu', 'eval_mode': True}
        if self.checkpoint:
            kwargs.update({'checkpoint_path': self.checkpoint, 'load_from_HF': False})
        self.model = build_sam3_image_model(**kwargs)
        self.model.eval()
        print('[SAM 3] Model loaded on CPU.')

    def segment(self, image, prompt, confidence_threshold, max_detections):
        from sam3.model.sam3_image_processor import Sam3Processor

        with self.lock:
            self.load()
            self.model.to(self.device)
            processor = Sam3Processor(self.model, confidence_threshold=confidence_threshold)
            autocast = contextlib.nullcontext()
            if self.device.type == 'cuda' and torch.cuda.is_bf16_supported():
                autocast = torch.autocast('cuda', dtype=torch.bfloat16)

            try:
                with torch.inference_mode(), autocast:
                    state = processor.set_image(image)
                    output = processor.set_text_prompt(state=state, prompt=prompt)
                    masks = output['masks'].detach().to('cpu').numpy()
                    scores = output['scores'].detach().float().to('cpu').numpy().reshape(-1)

                union_mask, order = combine_masks(
                    masks=masks,
                    scores=scores,
                    image_shape=(image.height, image.width),
                    confidence_threshold=confidence_threshold,
                    max_detections=max_detections,
                )

                mask_buffer = io.BytesIO()
                Image.fromarray(union_mask, mode='L').save(mask_buffer, format='PNG')
                return {
                    'mask': base64.b64encode(mask_buffer.getvalue()).decode('ascii'),
                    'detection_count': int(len(order)),
                    'scores': scores[order].tolist(),
                }
            finally:
                if not self.keep_gpu and self.model is not None:
                    self.model.to('cpu')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


class SAM3RequestHandler(BaseHTTPRequestHandler):
    runtime = None

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != '/health':
            self._send_json(404, {'error': 'Not found'})
            return
        self._send_json(200, {
            'status': 'ok',
            'device': str(self.runtime.device),
            'model_loaded': self.runtime.model is not None,
        })

    def do_POST(self):
        if self.path != '/segment':
            self._send_json(404, {'error': 'Not found'})
            return

        try:
            content_length = int(self.headers.get('Content-Length', '0'))
            if content_length <= 0 or content_length > 128 * 1024 * 1024:
                raise ValueError('Invalid request size.')
            payload = json.loads(self.rfile.read(content_length))
            image = Image.open(io.BytesIO(base64.b64decode(payload['image']))).convert('RGB')
            prompt = str(payload['prompt']).strip()
            if not prompt:
                raise ValueError('A text prompt is required.')

            result = self.runtime.segment(
                image=image,
                prompt=prompt,
                confidence_threshold=float(payload.get('confidence_threshold', 0.3)),
                max_detections=int(payload.get('max_detections', 0)),
            )
            self._send_json(200, result)
        except Exception as error:
            self._send_json(500, {'error': str(error)})

    def log_message(self, format, *args):
        print(f'[SAM 3] {self.address_string()} - {format % args}')


def main():
    parser = argparse.ArgumentParser(description='Local SAM 3 mask worker for Fooocus.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7866)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--keep-gpu', action='store_true')
    args = parser.parse_args()

    SAM3RequestHandler.runtime = SAM3Runtime(checkpoint=args.checkpoint, keep_gpu=args.keep_gpu)
    server = ThreadingHTTPServer((args.host, args.port), SAM3RequestHandler)
    print(f'[SAM 3] Worker listening on http://{args.host}:{args.port}')
    server.serve_forever()


if __name__ == '__main__':
    main()
