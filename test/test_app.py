# Using py.test framework
from service import Intro, Clf


def test_example_message(client):
    """Example message should be returned"""
    client.app.add_route('/pred', Intro())

    result = client.simulate_get('/pred')
    assert result.json == {
        'message': 'This service verifies a model using the WBC test data set. '
                   'Invoke using the form /pred/index of test sample>. For example, /pred/24'}, \
        "The service test will fail until a trained model has been approved"


def test_classification_request(client):
    """Expected classification for WBC sample should be returned"""
    client.app.add_route('/pred/{index:int(min=0)}', Clf())

    result = client.simulate_get('/pred/1')
    assert result.status == "200 OK", "The service test will fail until a trained model has been approved"
    assert all(k in result.json for k in (
        "index", "predicted_label", "predicted")), "The service test will fail until a trained model has been approved" 
