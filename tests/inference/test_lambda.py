import pytest

from jarvis2.inference.app import lambda_handler


@pytest.mark.skip(reason="This is a local test that uses a downloaded model.")
def test_lambda():
    out = lambda_handler(
        event={},
        context=None
    )

    assert out['statusCode'] == 200
