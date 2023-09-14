class Config:
    def __init__(self, max_features, learning_rate, batch_size, feature_number, seq_max_len, embedding_output, n_hidden,
                 n_classes, n_epochs, n_input=None, isseq=True, keep_prob=0.5, miu=1e-5, layers = 1):
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._feature_number = feature_number
        self._seq_max_len = seq_max_len
        self._embedding_output = embedding_output
        self._n_hidden = n_hidden
        self._n_classes = n_classes
        if n_input:
            self._n_input = n_input + 10 * embedding_output
        else:
            self._n_input = (feature_number - 3) + 3 * embedding_output
        self._max_features = max_features
        self._data_shape = [seq_max_len, feature_number]
        self._keep_prob = keep_prob
        self._miu = miu
        if isseq:
            self._label_shape = [seq_max_len, n_classes]
        else:
            self._label_shape = [n_classes]
        self._n_epochs = n_epochs
        self._layers = layers
        return

    @property
    def miu(self):
        return self._miu

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def feature_number(self):
        return self._feature_number

    @property
    def seq_max_len(self):
        return self._seq_max_len

    @property
    def embedding_output(self):
        return self._embedding_output

    @property
    def n_hidden(self):
        return self._n_hidden

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def n_input(self):
        return self._n_input

    @property
    def max_features(self):
        return self._max_features

    @property
    def data_shape(self):
        return self._data_shape

    @property
    def n_epochs(self):
        return self._n_epochs

    @property
    def label_shape(self):
        return self._label_shape
    
    @property
    def layers(self):
        return self._layers

    @property
    def keep_prob(self):
        return self._keep_prob

    def get_config(self):
      return {
          'miu': self.miu,
          'learning_rate': self.learning_rate,
          'batch_size': self.batch_size,
          'feature_number': self.feature_number,
          'seq_max_len': self.seq_max_len,
          'embedding_output': self.embedding_output,
          'n_hidden': self.n_hidden,
          'n_classes': self.n_classes,
          'n_input': self.n_input,
          'max_features': self.max_features,
          'data_shape': self.data_shape,
          'n_epochs': self.n_epochs,
          'label_shape': self.label_shape,
          'layers': self.layers,
          'keep_prob': self.keep_prob
      }