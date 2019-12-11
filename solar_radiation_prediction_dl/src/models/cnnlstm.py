import tensorflow as tf



class cnnLSTM(tf.keras.Model):
    """
        base cnn lstm model
    """

    def __init__(self, input_shape):
        super(cnnLSTM, self).__init__(name="")

        ##input layer
        self.input_ = tf.keras.layers.Input(shape=input_shape)

        ##cnn model 
        self.conv1 = tf.keras.layers.Conv1D(16, 3 , activation="relu")
        self.maxpool1 = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.flatten1 = tf.keras.layers.Flatten()

        # self.timedestributed = tf.keras.layers.TimeDistributed()

        ##lstm model
        self.lstm1 = tf.keras.layers.LSTM(64,activation='relu')
        self.dense1 = tf.keras.layers.Dense(1)


    def call(self,input_tensor):

        x = self.lstm1(input_tensor)
        x = self.dense1(x)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        

        return x

