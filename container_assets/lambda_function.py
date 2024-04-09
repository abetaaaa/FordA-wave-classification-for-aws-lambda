import tflite_runtime.interpreter as tflite
import numpy as np
import json


MODEL_PATH = './models/best_1d_cnn_model.tflite'


def lambda_handler(event, context):
    """lambda_handler

    Args:
        event (dic): e.g. {'items': [[0, 0, ...], [1, 1, ...], ...]}

    Returns:
        dic: statusCode and result
    """

    model = TFLiteModel(MODEL_PATH)

    input_list = event['items']
    input_np = np.array(input_list)

    if len(input_list) == 1:
        input_np = input_np.reshape(model.input_shape)

        result = model.predict(input_np)
        result_dic = {'result': result}

        print(result_dic)  # output to CloudWatch logs

        return {'statusCode': 200,
                'body': result_dic}

    else:
        results = []
        for batch_i in range(input_np.shape[0]):
            result = model.predict(input_np[batch_i, :, :])
            results.append(result)

        results_dic = {'result': results}
        print(results_dic)  # output to CloudWatch logs

        return {'statusCode': 200,
                'body': results_dic}


class TFLiteModel:
    """TensorFlow Lite Model
    """
    def __init__(self, tflite_model):
        # Load the TFLite model
        self._interpreter = tflite.Interpreter(model_path=tflite_model)

        # Allocate tensors
        self._interpreter.allocate_tensors()

        # Set input and output tensors
        self.input_details = self._interpreter.get_input_details()
        self.output_details = self._interpreter.get_output_details()

        # Set input shape
        self.input_shape = self.input_details[0]['shape']

    def predict(self, input_data):
        """Execute prediction

        Args:
            input_data (ndarray): data to be predicted

        Raises:
            HTTPException: Application Error

        Returns:
            str: result
        """
        # Execute prediction
        output_data = self.__predict(input_data)

        # Judge Positive/Negative
        result_number = int(np.argmax(output_data, axis=1))

        result = {'Class': result_number}

        return result

    def __predict(self, input_data):
        # Convert dtype from float64 to float32
        input_data = np.array(input_data, dtype='float32')

        # Set inputs
        self._interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Execute
        self._interpreter.invoke()

        output_data = self._interpreter.get_tensor(self.output_details[0]['index'])

        return output_data
