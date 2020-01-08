import json
import tensorflow as tf

class AttentionModule(tf.keras.layers.Layer):
    """
        This layer is the implementation of Simple History attention LSTM
    """
    
    def __init__(self, history):
        super(AttentionModule, self).__init__()
        self.history_size = history
        
    def build(self,input_shape):
        self.kernel = self.add_weight("kernel", shape=[self.history_size, self.history_size])
        
    def call(self, input_tensor):
        attended_weights = tf.nn.softmax(tf.nn.tanh(tf.matmul(self.kernel, input_tensor)))
        attended_input = tf.multiply(attended_weights, input_tensor)

        return attended_input

class lstmCNN(tf.keras.Model):
    """
        base cnn lstm model
    """

    def __init__(self, input_shape):
        super(lstmCNN, self).__init__(name="")
        self.input_shape_ = input_shape

        ##lstm model
        self.lstm1 = tf.keras.layers.LSTM(256,activation='tanh')
        # self.lstm2 = tf.keras.layers.LSTM(256,activation='tanh')

        #reshaping layers
        self.reshape1 = tf.keras.layers.Reshape((16 , 16))

        ##cnn model 
        self.conv1 = tf.keras.layers.Conv1D(64, 3 , activation="tanh")
        self.maxpool1 = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.conv2 = tf.keras.layers.Conv1D(128, 3 , activation="tanh")
        self.maxpool2 = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.flatten1 = tf.keras.layers.Flatten()
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dense1 = tf.keras.layers.Dense(2048)
        self.dense2 = tf.keras.layers.Dense(1024)
        self.dense3 = tf.keras.layers.Dense(1)
        self.attention = AttentionModule(self.input_shape_[-2])

    def call(self,input_tensor):
        
        x = self.lstm1(input_tensor)
        # x = self.lstm2(x)
        x = self.reshape1(x)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten1(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
      
        return x


