. ./awsconf.ini

find models -type f -name "*.tflite" -exec cp -f {} container_assets/models \;
cd container_assets

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
