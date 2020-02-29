class Pipeline:
    def __init__(self, pipes):
        self.pipes = pipes

    def __call__(self, *args, **kwargs):
        for pipe in self.pipes:
            args, kwargs = pipe(*args, **kwargs)
        return args, kwargs