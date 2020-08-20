# Import your handlers here
from service import Clf, Intro


# Configuration for web API implementation
def config(api):

    # Instantiate handlers and map routes
    api.add_route('/pred', Intro())
    api.add_route('/pred/{index:int(min=0)}', Clf())
