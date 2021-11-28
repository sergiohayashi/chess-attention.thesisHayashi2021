#
# Ref=> https://www.tensorflow.org/tutorials/text/image_captioning
# https://colab.research.google.com/notebooks/gpu.ipynb#scrollTo=sXnDmXR7RDr2
#
import json

import tensorflow as tf
import numpy as np

from config import config

device_name = tf.test.gpu_device_name()
print('Found GPU at: {}'.format(device_name))
print(tf.__version__)

device_name = tf.test.gpu_device_name()
print('Found GPU at: {}'.format(device_name))


# erro Blas GEMM launch failed quando usando tensorflow 2.4 INICIO
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# erro Blas GEMM launch failed quando usando tensorflow 2.4 FIM
# https://forums.developer.nvidia.com/t/could-not-create-cudnn-handle-cudnn-status-alloc-failed/108261
# precisa?
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        # print( 'BahdanauAttention.features =>', features.shape) #(64, 64, 256), segundo 64=length(encoder_output)
        # print( 'BahdanauAttention.hidden =>', hidden.shape) #(64, 512)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # print( 'BahdanauAttention.hidden_with_time_axis =>', hidden_with_time_axis.shape) #(64, 1, 512)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        # print( 'BahdanauAttention.score =>', score.shape)   #(64, 64, 512)

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        # print( 'BahdanauAttention.attention_weights =>', attention_weights.shape)  #(64, 64, 1) segundo 64 vem do tamanho da sequencia apos cnn

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        # print( 'BahdanauAttention.context_vector =>', context_vector.shape)  #(64, 64, 256) segundo 64= length(encoder_output)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # print( 'BahdanauAttention.context_vector =>', context_vector.shape) #(64, 256)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):

    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim, units):
        super(CNN_Encoder, self).__init__()
        self.units = units
        # shape after fc == (batch_size, 64, embedding_dim)
        # self.gru1= tf.keras.layers.GRU(self.units,
        #      dropout= ENCODER_DROPOUT,
        #  return_sequences=True)
        self.bgru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.units,
                                                                      dropout=0.2,
                                                                      return_sequences=True))
        self.drop = tf.keras.layers.Dropout(0.2)
        # self.gru= tf.keras.layers.GRU(self.units,
        #                            return_sequences=True)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.bgru(x)
        # x = self.bgru2( x)
        x = self.drop(x)
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       dropout=0.2,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        # self.gru.trainable= False
        self.fc1 = tf.keras.layers.Dense(self.units)  # 512
        # self.fc2 = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.0001))    #256
        self.fcF = tf.keras.layers.Dense(vocab_size)  # 120

        # self.bn0 = tf.keras.layers.BatchNormalization()
        # self.bn1 = tf.keras.layers.BatchNormalization()
        # self.bn2 = tf.keras.layers.BatchNormalization()

        # self.drop0= tf.keras.layers.Dropout( 0.7)
        self.drop = tf.keras.layers.Dropout(0.2)
        # self.drop2= tf.keras.layers.Dropout( 0.7)

        # self.dropout = tf.keras.layers.Dropout(0.5)
        # self.batchnormalization = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,
        #                     center=True, scale=True, beta_initializer='zeros',
        #                     gamma_initializer='ones', moving_mean_initializer='zeros',
        #                     moving_variance_initializer='ones', beta_regularizer=None,
        #                     gamma_regularizer=None, beta_constraint=None,
        #                     gamma_constraint=None)

        self.attention = BahdanauAttention(self.units)
        # if self.FREEZE_ENCODER:
        #     self.attention.trainable = False

    def freeze_attention(self):
        self.attention.trainable = False

    def call(self, x, features, hidden):
        # print( 'decoder.call=> ', x.shape, features.shape, hidden.shape) #(64, 1) (64, 64, 256) (64, 512)

        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)
        # print( 'decoder.attention=> ', context_vector.shape, attention_weights.shape) #(64, 256) (64, 64, 1)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # print( 'decoder.embedding=> ', x.shape) #(64, 1, 256)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # print( 'decoder.concat=> ', x.shape) #(64, 1, 512)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # output = self.bn0(output)
        # output = self.drop0(output)

        x = self.fc1(output)  # fc1= Dense( 512)
        # x= output
        # x = self.bn1(x)
        x = self.drop(x)

        x = tf.reshape(x, (-1, x.shape[2]))
        # x = self.fc2(x) # 256
        # x = self.bn2(x)
        # x = self.drop2(x)

        # x= self.dropout(x)
        # x= self.batchnormalization(x)
        x = self.fcF(x)  # fc2= Dense(5000)
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class AttentionEncoderDecoderModel:
    def __init__(self,
                 NUM_LINHAS=2,
                 NO_TEACH=True,
                 ):
        # -1 gera (None, 18, 21, 512)
        if NUM_LINHAS == 2:
            self.ATTENTION_SHAPE = (12, 53)
            self.INPUT_SHAPE = (200, 862)
        elif NUM_LINHAS == 8:
            self.ATTENTION_SHAPE = (50, 53)
            self.INPUT_SHAPE = (800, 862)
        else:
            raise NameError("Suporta somente 2 e 8 linhas")

        if "FORCE_INPUT_SIZE" in config:
            print("\nUsa FORCE_INPUT_SIZE informada em config.py", json.dumps(config["FORCE_INPUT_SIZE"]), "\n")
            self.ATTENTION_SHAPE = config["FORCE_INPUT_SIZE"]["ATTENTION_SHAPE"]
            self.INPUT_SHAPE = config["FORCE_INPUT_SIZE"]["INPUT_SHAPE"]

        self.FEATURES_SHAPE = 512
        self.ATTENTION_FEATURES_SHAPE = self.ATTENTION_SHAPE[0] * self.ATTENTION_SHAPE[1]  # 16*19   # 308
        self.EMBEDDING_DIM = 256
        self.UNITS = 512
        self.LEARNING_RATE = 0.0005
        self.NO_TEACH = NO_TEACH
        self.FREEZE_ENCODER = False
        self.TRAIN_LENGTH = 16
        # self = Config()
        self.tokenizer, self.VOCAB_SIZE = TokenizerBuilder().build()
        assert self.VOCAB_SIZE == 180, 'Tamanho do vocabulario não confere!'

        self.image_model = None
        self.encoder = None
        self.decoder = None
        self.loss_object = None
        self.image_features_extract_model = None
        self.steps = Steps(self);
        self.optimizer = None

    def build(self):
        self.image_model = tf.keras.applications.VGG16(include_top=False,
                                                       weights='imagenet',
                                                       input_shape=(self.INPUT_SHAPE[0], self.INPUT_SHAPE[1],
                                                                    3))  # => gera (16, 19, 2048)
        # input_shape= (900, 678, 3))  # => gera (16, 19, 2048)
        # O input shape nao é obrigatório, mas setando dá para
        # ver o tamanho do output
        new_input = self.image_model.input
        hidden_layer = self.image_model.layers[-2].output
        print("Shape da imagem ao final da CNN: ", self.image_model.layers[-2].output.shape)
        self.image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

        self.encoder = CNN_Encoder(self.EMBEDDING_DIM, self.UNITS)
        if self.FREEZE_ENCODER:
            self.encoder.trainable = False

        self.decoder = RNN_Decoder(self.EMBEDDING_DIM, self.UNITS, self.VOCAB_SIZE)

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE)
        return self

    def print_summary(self):
        print(self.image_model.summary())
        try:
            if self.encoder and 'summary' in dir(self.encoder):
                print(self.encoder.summary())
            if self.decoder and 'summary' in dir(self.decoder):
                print(self.decoder.summary())
        except:
            pass
            # print("Unexpected error:", sys.exc_info()[0])

    def loss_function(self, real, pred):
        # print( 'loss_function.real, pred', real.shape, pred.shape)  #(64,) (64, 5001)
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)


class Steps:
    def __init__(self, model):
        self.model = model
        self.ckpt = None

        # metricas
        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.valid_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # @staticmethod
    def load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.model.INPUT_SHAPE)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img, image_path

    def saveCheckpointTo(self, checkpointRelativePath):
        if self.ckpt is None:
            self.ckpt = tf.train.Checkpoint(encoder=self.model.encoder,
                                            decoder=self.model.decoder,
                                            optimizer=self.model.optimizer)
        ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpointRelativePath, max_to_keep=1)
        ckpt_manager.save()
        print('saved to ', ckpt_manager.latest_checkpoint)

    def restoreFromLatestCheckpoint(self, checkpointPath):
        self.ckpt = tf.train.Checkpoint(encoder=self.model.encoder,
                                        decoder=self.model.decoder,
                                        optimizer=self.model.optimizer)

        latest = tf.train.latest_checkpoint(checkpointPath)
        if latest:
            print("restore from pretraining  " + latest, '...')
            self.ckpt.restore(tf.train.latest_checkpoint(checkpointPath))
        else:
            print('no checkpoint found in ', checkpointPath)

    @staticmethod
    def checkpointExists(checkpointPath):
        latest = tf.train.latest_checkpoint(checkpointPath)
        return latest

    def evaluate(self, image, _length=4):
        # print( 'evaluate>>')
        # print( "_length", _length)

        attention_plot = np.zeros((32, self.model.ATTENTION_FEATURES_SHAPE))

        hidden = self.model.decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(self.load_image(image)[0], 0)
        img_tensor_val = self.model.image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = self.model.encoder(img_tensor_val)

        dec_input = tf.expand_dims([0 if self.model.NO_TEACH else self.model.tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(_length):
            predictions, hidden, attention_weights = self.model.decoder(dec_input, features, hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            result.append(self.model.tokenizer.index_word[predicted_id] if predicted_id < len(
                self.model.tokenizer.index_word) else "OUT")

            dec_input = tf.expand_dims([0 if self.model.NO_TEACH else predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot, None

    @tf.function
    def train_step(self, img_tensor, target, train_length):
        loss = 0
        zeros = np.zeros(target.shape[0]).astype(int)

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = self.model.decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims(
            zeros if self.model.NO_TEACH else [self.model.tokenizer.word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = self.model.encoder(img_tensor)

            for i in range(1, train_length + 1):
                predictions, hidden, _ = self.model.decoder(dec_input, features, hidden)

                # tf.keras.backend.print_tensor( predictions)
                # tf.keras.backend.print_tensor( tf.reduce_max( predictions))
                # tf.keras.backend.print_tensor( tf.reduce_min( predictions))
                loss += self.model.loss_function(target[:, i], predictions)
                self.train_acc_metric.update_state(target[:, i], predictions)

                dec_input = tf.expand_dims(zeros if self.model.NO_TEACH else target[:, i], 1)

        total_loss = (loss / int(train_length))

        # update model
        trainable_variables = self.model.encoder.trainable_variables + self.model.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    #
    # call train without update the gradient
    #
    @tf.function
    def test_step(self, img_tensor, target, train_length):
        loss = 0
        zeros = np.zeros(target.shape[0]).astype(int)

        hidden = self.model.decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims(
            zeros if self.model.NO_TEACH else [self.model.tokenizer.word_index['<start>']] * target.shape[0], 1)
        features = self.model.encoder(img_tensor)

        for i in range(1, train_length + 1):
            predictions, hidden, _ = self.model.decoder(dec_input, features, hidden)

            loss += self.model.loss_function(target[:, i], predictions)
            self.valid_acc_metric.update_state(target[:, i], predictions)

            dec_input = tf.expand_dims(zeros if self.model.NO_TEACH else target[:, i], 1)

        total_loss = (loss / int(train_length))
        return loss, total_loss


class TokenizerBuilder:

    def __init__(self):
        pass

    @staticmethod
    def build_tokenizer():
        with open(config['VOCAB_FILE']) as file:
            labels = [line.strip() for line in file]
        labels = ['<start> ' + label + ' <end>' for label in labels]

        # Choose the top 5000 words from the vocabulary
        print('building...')
        top_k = 5000  # para ajustar ao modelo antigo...
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                          lower=False,
                                                          oov_token="<unk>",
                                                          filters=' ')
        # forca a usar sempre uma lista com todas as words com 1 ocorrecia de cada word
        tokenizer.fit_on_texts(labels)
        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'
        return tokenizer

    def build(self):
        tokenizer = self.build_tokenizer()
        print('total do vocabulario= ', len(tokenizer.word_index))  # expected 1578

        VOCAB_SIZE = len(tokenizer.word_index) + 1
        print('VOCAB_SIZE', VOCAB_SIZE)
        return tokenizer, VOCAB_SIZE
