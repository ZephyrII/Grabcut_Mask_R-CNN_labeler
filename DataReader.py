class DataReader:

    def __init__(self, data_path):
        self.data_path = data_path
        self.frame_no = None
        self.frame = None
        self.frame_shape = None

    def next_frame(self):
        pass

    def forward_n_frames(self, n):
        pass

    def backward_n_frames(self, n):
        pass
