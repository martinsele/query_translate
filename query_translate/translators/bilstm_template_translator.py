import time
from functools import partial
from typing import Iterable

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

from query_translate.generator.dynamic_utterance_data_manager import DynamicUtteranceDataManager
from query_translate.model.tf_bilstm import Encoder, Decoder
from query_translate.model.tf_transformer import TfUtils
from query_translate.utils.type_defs import DynamicDataSample, DynTemplateFieldsEnum


class BiLSTMTemplateTranslator:
    """Translator of NL queries to DynamicTemplate types using Bi-LSTM model."""

    _dataset: Iterable[DynamicDataSample]

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def __init__(self):
        self.in_tokens_file = './checkpoints/lstm_in_tokens_'
        self.out_tokens_file = './checkpoints/lstm_out_tokens_'
        self.checkpoint_path = "./checkpoints/train_lstm"
        self.max_length = 50
        self.embedding_dim = 256
        self.units = 1024

    def train_model(self):
        batch_size = 64
        epochs = 20

        manager = DynamicUtteranceDataManager(generate_data=True)
        dataset = partial(self.dataset_generator, manager.generate_data_random(), batch_size)
        train_dataset = tf.data.Dataset.from_generator(dataset, output_types=(tf.string, tf.string))

        self.tokenizer_utter = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (t_in.numpy() for t_in, t_out in train_dataset), target_vocab_size=2 ** 10 + 2)
        self.tokenizer_transl = tfds.features.text.TokenTextEncoder(vocab_list=[e.value for e in DynTemplateFieldsEnum])
        self.tokenizer_utter.save_to_file(self.in_tokens_file)
        self.tokenizer_transl.save_to_file(self.out_tokens_file)

        encoder = Encoder(self.tokenizer_utter.vocab_size+2, self.embedding_dim, self.units, batch_size)    # +2 for start/end tokens
        decoder = Decoder(self.tokenizer_transl.vocab_size+2, self.embedding_dim, self.units, batch_size)

        optimizer = tf.keras.optimizers.Adam()
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

        train_dataset = self.prepare_data(train_dataset, self.tokenizer_utter, self.tokenizer_transl,
                                          prefetch=True, batch_size=batch_size)

        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        tf.config.experimental_run_functions_eagerly(True)

        @tf.function
        def train_step(inp, targ, enc_hidden):
            loss = 0
            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, enc_hidden)
                dec_hidden = enc_hidden

                start_token = [self.tokenizer_transl.vocab_size]
                dec_input = tf.expand_dims(start_token * batch_size, 1)

                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                    loss += self.loss_function(targ[:, t], predictions)
                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)

                    train_accuracy(targ[:, t], predictions)

            batch_loss = (loss / int(targ.shape[1]))
            variables = encoder.trainable_variables + decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss

        for epoch in range(epochs):
            start = time.time()
            train_accuracy.reset_states()

            enc_hidden = encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(train_dataset):
                batch_loss = train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, batch_loss.numpy(), train_accuracy.result()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix=self.checkpoint_path)

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                total_loss / (batch*batch_size),
                                                                train_accuracy.result()))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def evaluate(self, sentence, encoder: Encoder, decoder: Decoder):
        attention_plot = np.zeros((self.max_length, self.max_length))

        # pad sentence to max_length
        inputs = [self.tokenizer_utter.vocab_size] + self.tokenizer_utter.encode(sentence) + \
                 [self.tokenizer_utter.vocab_size + 1]
        inputs = inputs + [0] * (self.max_length - len(inputs))

        result = ''
        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([[self.tokenizer_transl.vocab_size]], 0)

        for t in range(self.max_length):
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()
            result += self.tokenizer_transl.decode(predicted_id) + ' '

            if predicted_id == self.tokenizer_transl.vocab_size + 1:
                return result, sentence, attention_plot

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)
        return result, sentence, attention_plot

    def translate(self, sentence, encoder: Encoder, decoder: Decoder):
        result, sentence, _ = self.evaluate(sentence, encoder, decoder)
        print('Input: %s' % sentence)
        print('Predicted translation: {}'.format(result))

    def prepare_data(self, dataset, tokenizer_in, tokenizer_out,  prefetch: bool = True, batch_size: int = 32):
        """ Prepare data - encode and add start/end tokens """

        def encode(t_in, t_out):
            lang1 = [tokenizer_in.vocab_size] + tokenizer_in.encode(t_in.numpy()) + [tokenizer_in.vocab_size + 1]
            lang2 = [tokenizer_out.vocab_size] + tokenizer_out.encode(t_out.numpy()) + [tokenizer_out.vocab_size + 1]
            return lang1, lang2

        def tf_encode(t_in, t_out):
            result_in, result_out = tf.py_function(encode, [t_in, t_out], [tf.int64, tf.int64])
            result_in.set_shape([None])
            result_out.set_shape([None])
            return result_in, result_out

        buffer_size = 200
        if prefetch:
            preprocessed = (dataset.map(tf_encode).filter(TfUtils.filter_max_length).cache().shuffle(buffer_size))
            tf_dataset = (preprocessed.padded_batch(batch_size, padded_shapes=([None], [None])).prefetch(
                          tf.data.experimental.AUTOTUNE))
        else:
            preprocessed = (dataset.map(tf_encode).filter(TfUtils.filter_max_length))
            tf_dataset = (preprocessed.padded_batch(batch_size, padded_shapes=([None], [None])))
        return tf_dataset

    @staticmethod
    def dataset_generator(data_set, batch_size: int = 64):
        for idx, sample in enumerate(data_set):
            input_ = sample.utterance
            output = DynamicUtteranceDataManager.template_translation_to_text(sample.translation)
            if idx == batch_size*300 - 1:
                return input_, output
            yield input_, output

    @classmethod
    def loss_function(cls, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = cls.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)


if __name__ == "__main__":
    classifier = BiLSTMTemplateTranslator()
    classifier.train_model()
