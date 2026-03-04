class DownloadError(Exception):
    def __init__(self, code, url):
        super().__init__(f"Error: {code} while downloading: {url}")
        self.code = code
        self.url = url