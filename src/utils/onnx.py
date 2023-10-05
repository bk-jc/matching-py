import os

import torch


def export_to_onnx(a, model, test_ds, version):
    def get_example_input(ds):
        return {
            "cv": [ds.dataset[0]['cv']],
            "job": [ds.dataset[0]['job']]
        }

    _ = model(**get_example_input(test_ds))
    # Export the model to ONNX
    example_input = get_example_input(test_ds)
    onnx_path = os.path.join(a.save_path, a.exp_name, version, "jarvis_v2.onnx")
    os.makedirs(os.path.join(a.save_path, a.exp_name, version), exist_ok=True)
    torch.onnx.export(model, tuple(example_input.values()), onnx_path,
                      input_names=list(example_input.keys()), output_names=["similarity"])
