import os

from src.train import main, parse_args


def test_train(tmpdir):
    main(parse_args([
        "--save_path", str(tmpdir)
    ]))
    out_dir = os.path.join(tmpdir, parse_args([]).exp_name)
    assert os.listdir(out_dir)[0].endswith(".onnx")
