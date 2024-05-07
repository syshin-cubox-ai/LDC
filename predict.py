import argparse
import os
import platform
import time

import cv2
import numpy as np
import onnxruntime


class LDC:
    def __init__(self, model_path: str, device: str):
        """
        Args:
            model_path: Model file path.
            device: Device to inference.
        """
        assert os.path.exists(model_path), f'model_path is not exists: {model_path}'
        assert device in ['cpu', 'cuda', 'openvino', 'tensorrt'], f'device is invalid: {device}'

        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        elif device == 'cuda':
            providers = ['CUDAExecutionProvider']
        elif device == 'openvino':
            providers = ['OpenVINOExecutionProvider']
        elif device == 'tensorrt':
            providers = ['TensorrtExecutionProvider']
        else:
            raise ValueError(f'device is invalid: {device}')
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        session_input = self.session.get_inputs()[0]
        self.input_name = session_input.name

    def _transform_image(self, img: np.ndarray) -> np.ndarray:
        """
        This performs BGR to RGB, HWC to CHW, normalization, and adding batch dimension.
        (mean=(123.68, 116.779, 103.939), std=(1, 1, 1))
        """
        assert img.shape[0] % 8 == 0 and img.shape[1] % 8 == 0, 'Image must be divisible by 2^3=8'

        img = cv2.dnn.blobFromImage(img, 1, img.shape[:2][::-1], (123.68, 116.779, 103.939), swapRB=True)
        return img

    def detect_one(self, img: np.ndarray) -> np.ndarray:
        img = self._transform_image(img)
        pred = self.session.run(None, {self.input_name: img})[0]
        return pred


# Global parameters
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='model file path.')
    parser.add_argument('--source', type=str, default='0', help='file/dir/webcam')
    parser.add_argument('--device', type=str, default='cuda', help='[cpu, cuda, openvino, tensorrt]')
    parser.add_argument('--draw-fps', action='store_true', help='Draw fps on the frame.')
    args = parser.parse_args()
    print(args)

    # Load detector
    detector = LDC(args.model_path, args.device)

    # Inference
    # source: webcam or video
    if args.source.isnumeric() or args.source.lower().endswith(VID_FORMATS):
        if args.source.isnumeric():
            if platform.system() == 'Windows':
                cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(int(args.source))
        else:
            cap = cv2.VideoCapture(args.source)
        assert cap.isOpened()

        if args.source.isnumeric():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        count = 1
        accumulated_time = 0
        fps = 0
        while cv2.waitKey(5) != ord('q'):
            # Load image
            ret, img = cap.read()
            assert ret, 'no frame has been grabbed.'

            # Detect edge
            start = time.perf_counter()
            pred = detector.detect_one(img)
            accumulated_time += (time.perf_counter() - start)
            if count % 10 == 0:
                fps = 1 / (accumulated_time / 10)
                accumulated_time = 0
            count += 1

            # Draw FPS
            if args.draw_fps:
                cv2.putText(pred, f'{fps:.2f} FPS', (10, 26), cv2.FONT_HERSHEY_DUPLEX,
                            0.66, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(pred, f'{fps:.2f} FPS', (10, 26), cv2.FONT_HERSHEY_DUPLEX,
                            0.66, (255, 255, 255), 1, cv2.LINE_AA)

            # Show prediction
            cv2.imshow('Edge Detection', pred)

        print('Quit inference.')
        cap.release()
        cv2.destroyAllWindows()

    # source: image
    elif args.source.lower().endswith(IMG_FORMATS):
        assert os.path.exists(args.source), f'Image not found: {args.source}'

        # Load image
        img: np.ndarray = cv2.imread(args.source)
        assert img is not None

        # Detect edge
        pred = detector.detect_one(img)

        # Save prediction
        cv2.imwrite('result.jpg', pred)
        print('Save result to "result.jpg"')
    else:
        raise ValueError(f'Wrong source: {args.source}')
