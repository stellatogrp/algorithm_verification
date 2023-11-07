class Step(object):

    """Docstring for Step. """

    def __init__(self, start_canon=None):
        """TODO: to be defined. """
        if start_canon is None:
            self.start_canon = 1
        else:
            self.start_canon = start_canon
        self.is_linstep = False

    def get_output_var(self):
        raise NotImplementedError
