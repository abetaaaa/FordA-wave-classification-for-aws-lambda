import argparse
import configparser
import json
import numpy as np
import random
import requests
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('env', help='test environment(aws/local_wsl/local_win)')
args = parser.parse_args()
env = args.env


class EnvParser(configparser.ConfigParser):
    """EnvParser
    """
    def _read(self, fp, fpname):
        def addsection(fp):
            yield "[DEFAULT]"
            yield from fp
        super(EnvParser, self)._read(addsection(fp), fpname)


def readucr(filename):
    """readucr
    read forda dataset

    Args:
        filename (str): specify .tsv file

    Returns:
        ndarray: data and label
    """
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]   # label
    x = data[:, 1:]  # data
    return x, y.astype(int)


def request_api(json_data, env):
    """request_api

    Args:
        json_data (dic): data to send
        env (str): test environment

    Returns:
        str: response
    """
    if env == 'aws':
        # Read awsconf.ini
        config_ini = EnvParser()
        config_ini.read('awsconf.ini', encoding='utf-8')

        # Get region
        region = config_ini.get('DEFAULT', 'REGION')

        # Get API ID
        command = \
            '. ./awsconf.ini; '\
            'aws apigateway get-rest-apis ' \
            '--query "items[?name == \'$APIGW_NAME\'].id" ' \
            '--output text'

        output = subprocess.check_output(command, shell=True,
                                         universal_newlines=True)
        rest_api_id = output.strip()

        # API Gateway Host Name
        hostname = f'{rest_api_id}.execute-api.{region}.amazonaws.com'

        # Stage
        stage = '/v1'

        # API Path
        api_path = '/predict'

        url = f'https://{hostname}{stage}{api_path}'

    elif env == 'local':
        url = 'http://172.23.128.1:9000/2015-03-31/functions/function/invocations'
    elif env == 'local_win':
        url = 'http://localhost:9000/2015-03-31/functions/function/invocations'

    response = requests.post(url, data=json_data,
                             headers={"Content-Type": "application/json"})

    return response


if __name__ == '__main__':

    ROOT_URL = 'https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/'

    x_train, y_train = readucr(ROOT_URL + "FordA_TRAIN.tsv")
    x_test, y_test = readucr(ROOT_URL + "FordA_TEST.tsv")

    # reshape
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # count the classes
    num_classes = len(np.unique(y_train))

    # shaffle for training
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    # standarize
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    # test
    SEND_DATA_NO = random.randint(0, len(y_test.tolist()))
    json_data = json.dumps(
        {'items': x_test[SEND_DATA_NO:SEND_DATA_NO+1, :, 0].tolist()}
        )

    res = request_api(json_data, env)

    print("=== test ===")
    print(f"No.{SEND_DATA_NO}")
    print(f"True Label      : {y_test[SEND_DATA_NO]}")
    print(f"Predicted Label : {res.json()['body']['result']['Class']}")
