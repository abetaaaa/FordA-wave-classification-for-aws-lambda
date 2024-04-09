<div id="top">

<img src="https://img.shields.io/badge/Python-3776AB.svg?logo=python&logoColor=white">
<img src="https://img.shields.io/badge/-AWS%20Lambda-232F3E.svg?logo=awslambda&style=flat">
<img src="https://img.shields.io/badge/-%20ECR-232F3E.svg?logo=amazonecs&style=flat">
<img src="https://img.shields.io/badge/-%20API%20Gateway-232F3E.svg?logo=amazonapigateway&style=flat">
<img src="https://img.shields.io/badge/-Docker-2496ED.svg?logo=docker&logoColor=white">
<img src="https://img.shields.io/badge/-WSL2-E95420?logo=ubuntu&logoColor=white">
<img src="https://img.shields.io/badge/-TensorFlow%20Lite-FF6F00?logo=tensorflow&logoColor=white">
<img src="https://img.shields.io/badge/-FastAPI-009688.svg?&logo=fastapi&logoColor=white">
<img src="https://img.shields.io/badge/-pandas-150458.svg?&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/-Jupyter Notebook-F37626.svg?&logo=jupyter&logoColor=white">

</br>
</div>

# FordA Wave Classificaiton for AWS Lambda
This repository contains the necessary resources to deploy a TensorFlow Lite wave classification model trained on the FordA dataset to AWS Lambda using Docker and ECR.   
The included Jupyter Notebook in this repository demonstarates the creation of a TensorFlow Lite model (1D-CNN, RNN, LSTM, GRU) using the [FordA dataset](https://www.timeseriesclassification.com/description.php?Dataset=FordA).  


## Goals
Build an application in an AWS environment that utilizes AWS Lambda to invoke a TensorFlow Lite model for inference on data transmitted through API Gateway, returning the result.

### Architecture
![archtecture](https://github.com/abetaaaa/FordA-wave-classification-for-aws-lambda/assets/78013610/5932bef0-352f-490e-9642-da2969d027b0)


## Environment

| Language/Library   | Version |
| ------------------ | ------- | 
| Python             | 3.9.18  |
| tflite-runtime     | 2.7.0   |
| fastapi            | 0.110.1 |
| pandas             | 2.2.1   |

## Repository Structure
```
.
├── FordA_wave_classification.ipynb
├── README.md
├── awsconf.ini
├── build_and_deploy_ecr.sh
├── container_assets
│   ├── Dockerfile
│   ├── lambda_function.py
│   ├── models
│   │   ├── ...
│   └── requirements.txt
├── deploy_cloudformation.sh
├── lambda_test.py
├── models
│   ├── best_1d_cnn_model.keras
│   ├── best_1d_cnn_model.tflite
│   ├── ...
├── template.yml
└── update_ecr.sh
```

- `FordA_wave_classification.ipynb`: Jupyter notebook for model creation and conversion to TensorFlow Lite.
- `awsconf.ini`: AWS configuration files (You need to edit this file).
- `build_and_deploy_ecr.sh`: Shell script to build and deploy the Docker image to Amazon Elastic Container Registry (ECR).
- `container_assets`: 
    - `Dockerfile`: Contains instructions to build the Docker image.
    - `lambda_function.py`: Lambda function code for inference.
    - `models/`: Directory where .tflite trained models are copied.
    - `requirements.txt`: File listing the required Python packages.
- `deploy_cloudformation.sh`: Shell script to deploy API Gateway with CloudFormation.
- `lambda_test.py`: Testing code of lambda_function.  
- `models/`: Directory containing trained models in Keras and TensorFlow Lite formats.
- `templete.yml`: CloudFormation Stack template.
- `update_ecr.sh`: Shell script to update the ECR repository.


## How to Deploy the TFLite Model to AWS Lambda

To Deploy the TFLite Model to AWS Lambda, follow these steps:

1. **Create Wave Classification Model**  
    i. Build a wave classification model using the FordA dataset with TensorFlow.  
    ii. Convert the model to TensorFlow Lite format.
2. **Build Docker Container Image and Deploy AWS Lambda Function**  
    i. Develop a Lambda function in the local environment.  
    ii. Create and build a Dockerfile and deploy the container image to the AWS environment.
3. **Deploy API Gateway with CloudFormation**  
    i. Create a CloudFormation Stack Template.  
    ii. Execute the `aws cloudformation deploy` command from AWS CLI. 


### Prerequisites
Before getting started, ensure you have the following:
- Docker is installed on your local machine.
    - if you don't have, install from the following. 
     https://www.docker.com/
- IAM user can access to below sevices.
    - Elastic Container Registry (ECR)
    - Lambda
    - CloudFormation
- IAM user's credentials is registered in `config` and `credentials` as [default].


## 1. Create Wave Classification Model

Now we use FordA dataset to create wave classificaion models.

### FordA Dataset
The dataset used here is called FordA. Each timeseries data is the engine noise value measured by the motor sensor. There are two classes in the dataset: one that contains waveforms of system anomalies and another that contains normal systems.

<div align="right">
    <a href="https://www.timeseriesclassification.com/description.php?Dataset=FordA">Details »</a>
</div>

| Length | Number of Classes | Dimensions |
| :----: | :---------------: | :--------: |
| 500    | 1755              | 1          |

| Class          | Train Size | Test Size |
| :-------------:| :--------: | :-------: |
| 0 (Abnormal?)  | 1755       | 639       |
| 1 (Normal?)    | 1846       | 1320      |

e.g.  
![wave-example](https://github.com/abetaaaa/FordA-wave-classification-for-aws-lambda/assets/78013610/9289043a-b779-482f-9baa-61333f04379a)


### 1-i. Build a Model

Four TensorFlow models (1D-CNN, RNN, LSTM, GRU) are created in the Jupyter Notebook ([FordA_wave_classification.ipynb](FordA_wave_classification.ipynb)).  
We can easily create models by using `keras.layers`.

e.g. 1D CNN model
```python
def make_1d_cnn_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)
```

### 1-ii. Convert TensorFlow Model to TensorFlow Lite Model

We can easily convert `.keras` to `.tflite` by using `TFLiteConverter`.

```python
# Load Keras model
keras_model = tf.keras.models.load_model(f'./models/best_1d_cnn_model.keras')

# Create TensorFlow Lite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

# Save the converted model
with open(f'./models/best_1d_cnn_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## 2. Build Docker Container Image and Deploy AWS Lambda Function

### 2-i. Develop a Lambda function in the local environment.  

I implemented a `lambda_handler` that receives an N×500×1 array of data and returns N results.
- N: Number of data to be inferred  

The data to be inferred is concluded in the body['item'] of the HTTP request.  
`print` function is used to output results to CloudWatch Logs.

```python
def lambda_handler(event, context):
    """lambda handler
    """
    model = TFLiteModel(MODEL_PATH)

    body = json.loads(event['body'])
    input_list = body['items']
    input_np = np.array(input_list)

    ~~~

    results = []
    for batch_i in range(input_np.shape[0]):
        result = model.predict(input_np[batch_i, :, :])
        results.append(result)

    print(results)  # output to CloudWatch logs

    return {'stasuCode': 200,
            'body': results}
```

### 2-ii. Create and build a Dockerfile and deploy the container image to the AWS environment.

First, you have to edit `awsconf.ini`.
```
AWS_ACCOUNT_ID=[YOUR_ACCOUNT_ID]
REGION=[YOUR_REGION]
IMAGE_NAME=tflite_lambda
REPOSITORY_NAME=tflite_lambda_python39
FUNCTION_NAME=tflite-predict-wave-classification
IAM_ROLE_NAME=lambda-ex
APIGW_NAME=forda-wave-classification-apigw
STACK_NAME=forda-api-gateway-stack
```


After that, to build the Docker image using the provided Dockerfile and push the Docker Image to ECR, run `build_and_deploy_ecr.sh`.


build_and_deploy_ecr.sh
```sh
. ./awsconf.ini

# Build the Docker image from Dockerfile
docker build --platform linux/amd64 -t $IMAGE_NAME:test .

# Tagging the Docker image
docker tag $IMAGE_NAME:test $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:latest

# Deploying the image
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
aws ecr create-repository --repository-name $REPOSITORY_NAME --region $REGION --image-scanning-configuration scanOnPush=true --image-tag-mutability MUTABLE
docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:latest


# Create the execution role
aws iam create-role --role-name lambda-ex --assume-role-policy-document '{"Version": "2012-10-17","Statement": [{ "Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'

# Add permissions to the execution role
aws iam attach-role-policy --role-name lambda-ex --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Create Lambda function
aws lambda create-function \
  --function-name $FUNCTION_NAME \
  --package-type Image \
  --code ImageUri=$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:latest \
  --role arn:aws:iam::$AWS_ACCOUNT_ID:role/$IAM_ROLE_NAME

```

Dockerfile
```Dockerfile
FROM public.ecr.aws/lambda/python:3.9

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN python3.9 -m pip install -r requirements.txt -t .

# Copy models
COPY models ${LAMBDA_TASK_ROOT}/models

# Copy funcion code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.lambda_handler" ]

```


## 3. Deploy API Gateway with CloudFormation

### 3-i. Create a CloudFormation Stack Template.  
Create a stack templete to deploy API Gateway using CloudFormation.  
Create an APIGateway for a REST API and integrate it with a Lambda function and ensure to grant permissions to the API Gateway using AWS::Lambda::Permission for access.

<div align="right">
    <a href="template.yml">template.yml »</a>
</div>

### 3-ii. Execute the `aws cloudformation deploy` command from AWS CLI. 
You just do to run `deploy_cloudformation.sh`.  
 An API Gateway is automatically created based on the stack template definition and integrated with the Lambda functions.

```sh
. ./awsconf.ini

aws cloudformation deploy --template template.yml --stack-name $STACK_NAME --parameter-overrides FunctionName=$FUNCTION_NAME ApiGwName=$APIGW_NAME
```

## Tests
Run `lambda_test.py` allows you to conduct testing of lambda_function.  
There are three options available:
- `aws`: This option allows you to conduct testing in your AWS environment.
- `local`: This option allows you to conduct testing on your local machine using WSL (Windows Subsystem for Linux).
- `local_win`: This option allows you to conduct testing on your local Windows machine.


When you run lambda_test.py, it randomly selects one data from the test data, performs inference on the Lambda function, and displays the result in the output.
```
$ python lambda_test.py aws 
=== test ===
No.785
True Label      : 1
Predicted Label : 1
```

## References
- [FordA Dataset](): Information about the FordA dataset used for training the model.
- [Timeseries classification from scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/): Example showing how to do timeseries classification form scratch using keras and FordA dataset.
