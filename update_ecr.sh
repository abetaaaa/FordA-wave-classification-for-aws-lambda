# Remove the old image
docker rm $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:latest

# Run this script when you have made any changes to the Docker image or lambda_function.py.
. ./awsconf.ini

find models -type f -name "*.tflite" -exec cp -f {} container_assets/models \;
cd container_assets

# Build the Docker image from Dockerfile
docker build --platform linux/amd64 -t $IMAGE_NAME:test .

# Tagging the Docker image
docker tag $IMAGE_NAME:test $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:latest

# Deploying the image
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:latest
