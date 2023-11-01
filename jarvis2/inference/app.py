import json

from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-multilingual-cased")


def lambda_handler(event, context):
    unmasker = pipeline('fill-mask', model='distilbert-base-multilingual-cased')
    tokens = unmasker("Hello I'm a [MASK] model.")

    return {
        "statusCode": 200,
        "body": json.dumps({k: json.dumps(v) for k, v in tokens[0].items()})
    }
