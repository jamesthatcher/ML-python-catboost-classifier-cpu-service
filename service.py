#!/usr/bin/python

import json
import time

import falcon
import numpy as np
import onnxruntime as rt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

PORT_NUMBER = 8080
start = time.time()

# instantiate the scaler
scaler = StandardScaler()

# get test data
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
labels = breast_cancer.target_names
X_count = X.shape[0]

end = time.time()
print("Loading time: {0:f} secs)".format(end - start))


# API Handler for WBC classifier
class Clf(object):
    """Handle classification requests for WBC dataset by ID"""

    def __init__(self):
        self.clf = rt.InferenceSession("model.onnx")
        self.input_name = self.clf.get_inputs()[0].name
        self.label_name = self.clf.get_outputs()[0].name

    def on_get(self, req, resp, index):
        if index < X_count:
            y_pred = self.clf.run([self.label_name], {self.input_name: X.astype(
                np.float32)})[0]
            payload = {'index': index, 'predicted_label': list(labels)[y_pred[0]], 'predicted': int(y_pred[0])}
            resp.body = json.dumps(payload)
            resp.status = falcon.HTTP_200
        else:
            raise falcon.HTTPBadRequest(
                "Index Out of Range. ",
                "The requested index must be between 0 and {:d}, inclusive.".format(X_count - 1)
            )


# API Handler for example message
class Intro(object):
    """Example of invoking the endpoint for classifying the Wincosin Breast Cancer dataset"""

    def on_get(self, req, resp):
        resp.body = '{"message": \
                    "This service verifies a model using the WBC test data set. Invoke using the form /pred/index of ' \
                    'test sample>. For example, /pred/24"}'
        resp.status = falcon.HTTP_200

