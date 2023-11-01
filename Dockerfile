FROM public.ecr.aws/lambda/python:3.9 as build

RUN yum install gcc -y

RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install torch==2.0.0 -f https://download.pytorch.org/whl/cpu/torch

COPY requirements.txt ./
RUN python3.9 -m pip install -r requirements.txt

COPY model_tokenizer ./model_tokenizer
COPY model.onnx .

COPY jarvis2/inference/app.py ./

FROM build AS test

# Smoketest Lambda function
COPY _test_lambda.py ./
RUN python3.9 _test_lambda.py

FROM build AS inference

CMD ["app.lambda_handler"]