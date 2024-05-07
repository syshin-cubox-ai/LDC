import os

import onnx
import onnx.shape_inference
import onnxsim
import ptflops
import torch

from modelB4 import LDC


def convert_onnx(model, img, output_path, opset=17, dynamic=False, simplify=True):
    assert isinstance(model, torch.nn.Module)

    model.eval()
    print("\n[in progress] torch.onnx.export...")

    # Define input and output names
    input_names = ['image']
    output_names = ['output']

    # Define dynamic_axes
    if dynamic:
        dynamic_axes = {input_names[0]: {2: 'H', 3: 'W'},
                        output_names[0]: {0: 'H', 1: 'W'}}
    else:
        dynamic_axes = None

    # Export model into ONNX format
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        model,
        img,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        dynamic_axes=dynamic_axes,
    )

    # Check exported onnx model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model, full_check=True)
    onnx.shape_inference.infer_shapes_path(output_path, output_path, check_type=True, strict_mode=True, data_prop=True)

    # Simplify ONNX model
    if simplify:
        model = onnx.load(output_path)
        input_shapes = {model.graph.input[0].name: img.shape}
        model, check = onnxsim.simplify(model, test_input_shapes=input_shapes)
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(model, output_path)
    print(f'Successfully export ONNX model: {output_path}')


if __name__ == '__main__':
    model = LDC()
    model.load_state_dict(torch.load('checkpoints/BRIND/11/11_model.pth'))
    img = torch.randn((1, 3, 640, 640))
    convert_onnx(model, img, 'onnx_files/LDC_B4.onnx')

    macs, params = ptflops.get_model_complexity_info(model, tuple(img.shape[1:]), False, False)
    print(f'ptflops GFLOPs: {macs * 2 / 1e9:.3f}, Params: {params / 1e6:.3f}M')
