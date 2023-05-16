from histomicstk.utils.girder_convenience_utils import connect_to_api


class StyxAPI(object):
    """ Params related to Styx DSA Instance """

    url = 'https://styx.neurology.emory.edu/girder/'
    apiurl = url + 'api/v1/'
    apikey = 'ReplaceMeWithYourAPIKey'

    @classmethod
    def connect_to_styx(cls, interactive=False):
        """Connect to styx API."""
        return connect_to_api(
            apiurl=cls.apiurl, apikey=cls.apikey, interactive=interactive)

    def __init__(self, interactive=False):
        self.gc = self.connect_to_styx(interactive=interactive)
