. ./awsconf.ini

aws cloudformation deploy --template template.yml --stack-name $STACK_NAME --parameter-overrides FunctionName=$FUNCTION_NAME ApiGwName=$APIGW_NAME