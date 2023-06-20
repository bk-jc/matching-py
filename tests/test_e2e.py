from src.train import main, parse_args


def test_train(tmpdir):
    main(parse_args([
        "--save_path", str(tmpdir)
    ]))
    assert tmpdir.listdir()[0].ext == ".onnx"
