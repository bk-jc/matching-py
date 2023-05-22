from app import lambda_handler


def test_lambda():
    out = lambda_handler(
        event={},
        context=None
    )

    assert out['statusCode'] == 200
