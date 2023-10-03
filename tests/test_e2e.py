import os

from src.train import main, parse_args


def test_train(tmpdir):
    main(parse_args([
        "--save_path", str(tmpdir)
    ]))
    out_dir = os.path.join(tmpdir, parse_args([]).exp_name, "output")
    assert len(os.listdir(out_dir)) == 1  # Only one output experiment
    out_time = os.listdir(out_dir)[0]
    out_dir = os.path.join(out_dir, out_time)
    assert os.listdir(out_dir)[0].endswith(".onnx")
