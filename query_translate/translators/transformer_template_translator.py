import time
from functools import partial
import random
from typing import Iterable

import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

from query_translate.generator.dynamic_utterance_data_manager import DynamicUtteranceDataManager
from query_translate.model.tf_transformer import TfTransformer, CustomSchedule
from query_translate.model.tf_utils import TfUtils
from query_translate.utils.type_defs import DynamicDataSample, DynTemplateFieldsEnum


class TransformerTemplateTranslator:
    """Translator of NL queries to DynamicTemplate types using Transformer model."""

    transformer: TfTransformer
    _dataset: Iterable[DynamicDataSample]

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def __init__(self):
        self.num_heads = 5
        self.d_model = self.num_heads * 50
        self.max_length = 50
        self.transformer = TfTransformer(num_layers=2, d_model=self.d_model, num_heads=self.num_heads, dff=512,
                                         input_vocab_size=2 ** 12 + 2, target_vocab_size=10,
                                         pe_input=self.max_length, pe_target=self.max_length)
        self.in_tokens_file = './checkpoints/in_tokens_'
        self.out_tokens_file = './checkpoints/out_tokens_'
        self.checkpoint_path = "./checkpoints/train"

    def save_dataset(self):
        manager = DynamicUtteranceDataManager(generate_data=True)
        manager.save_dataset()

    def train_model(self):
        batch_size = 64
        epochs = 50

        dataset = partial(self.dataset_generator, DynamicUtteranceDataManager.generate_data_random())
        train_dataset = tf.data.Dataset.from_generator(dataset, output_types=(tf.string, tf.string))

        self.tokenizer_utter = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (t_in.numpy() for t_in, t_out in train_dataset), target_vocab_size=2 ** 12)
        self.tokenizer_transl = tfds.features.text.TokenTextEncoder(vocab_list=[e.value for e in DynTemplateFieldsEnum])

        self.tokenizer_utter.save_to_file(self.in_tokens_file)
        self.tokenizer_transl.save_to_file(self.out_tokens_file)

        train_dataset = self.prepare_data(train_dataset, self.tokenizer_utter, self.tokenizer_transl,
                                          prefetch=True, batch_size=batch_size)

        learning_rate = CustomSchedule(self.d_model)
        optimizer = tf.keras.optimizers.SGD(learning_rate)
        # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        ckpt = tf.train.Checkpoint(transformer=self.transformer, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                                tf.TensorSpec(shape=(None, None), dtype=tf.int64)]

        tf.config.experimental_run_functions_eagerly(True)

        @tf.function(input_signature=train_step_signature)
        def train_step(inp, tar):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar_inp)

            with tf.GradientTape() as tape:
                predictions, _ = self.transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
                loss = self.loss_function(tar_real, predictions)

            gradients = tape.gradient(loss, self.transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

            train_loss(loss)
            self.update_accuracy(tar_real, predictions, train_accuracy)

        for epoch in range(epochs):
            start = time.time()
            train_loss.reset_states()
            train_accuracy.reset_states()

            # inp -> utterance, tar -> translation
            for (batch, (t_in, t_out)) in enumerate(train_dataset):
                train_step(t_in, t_out)

                if batch % 50 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, train_loss.result(), train_accuracy.result()))

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

            self.show_results_for_epoch()

            print(
                'Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
            # self.validation_error(val_dataset)
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    def prepare_data(self, dataset, tokenizer_in, tokenizer_out,  prefetch: bool = True, batch_size: int = 32):

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

    def validation_error(self, valid_dataset):
        correct_transl = 0
        samples = 0
        for (batch, (t_in, t_out)) in enumerate(valid_dataset):
            for i in range(t_in.shape[0]):
                samples += 1
                utt = self.tokenizer_utter.decode([idx for idx in t_in[i, :] if idx < self.tokenizer_utter.vocab_size])
                pred, _ = self.evaluate(utt)    # first try to decode
                min_dim = min(len(t_out[i, :]), len(pred))
                eqls = tf.reduce_all(tf.math.equal(tf.dtypes.cast(pred[:min_dim], tf.int32),
                                                   tf.dtypes.cast(t_out[i, :min_dim], tf.int32)))
                if eqls:
                    correct_transl += 1
        print(f"Validation accuracy: {(correct_transl/samples):.4f}")

    @staticmethod
    def dataset_generator(data_set, random_state=101):
        random.seed(random_state)
        for idx, sample in enumerate(data_set):
            input_ = sample.utterance
            output = DynamicUtteranceDataManager.template_translation_to_text(sample.translation)
            if idx > 100000:
                return input_, output
            yield input_, output

    def show_results_for_epoch(self, random_state=102):
        random.seed(random_state)
        for idx, sample in enumerate(DynamicUtteranceDataManager.generate_data_random()):
            self.translate(sample.utterance, plots=[])  #  "decoder_layer2_block2"
            if idx > 9:
                break

    def evaluate(self, inp_sentence):
        start_token = [self.tokenizer_utter.vocab_size]
        end_token = [self.tokenizer_utter.vocab_size + 1]

        # inp sentence is portuguese, hence adding the start and end token
        inp_sentence = start_token + self.tokenizer_utter.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # as the target is TemplateTranslation, the first word to the transformer should be the translation start token.
        decoder_input = [self.tokenizer_transl.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for _ in inp_sentence:
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(encoder_input, output, False, enc_padding_mask,
                                                              combined_mask, dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == self.tokenizer_transl.vocab_size + 1:
                return tf.squeeze(output, axis=0), attention_weights

            # concatentate the predicted_id to the output which is given to the decoder as its input.
            output = tf.concat([output, predicted_id], axis=-1)
        # noinspection PyUnboundLocalVariable
        return tf.squeeze(output, axis=0), attention_weights

    def evaluate_model(self):
        manager = DynamicUtteranceDataManager(generate_data=True)
        self._dataset = manager.dataset
        _, test = DynamicUtteranceDataManager.train_test_utter_split(manager.dataset, test_ratio=.1)
        self.tokenizer_utter = tfds.features.text.SubwordTextEncoder.load_from_file(self.in_tokens_file)
        self.tokenizer_transl = tfds.features.text.TokenTextEncoder.load_from_file(self.out_tokens_file)

        test_gen = partial(self.dataset_generator, test)
        test_dataset = tf.data.Dataset.from_generator(test_gen, output_types=(tf.string, tf.string))

        val_dataset = self.prepare_data(test_dataset, self.tokenizer_utter, self.tokenizer_transl, prefetch=False)

        ckpt = tf.train.Checkpoint(transformer=self.transformer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        # for i, sample in enumerate(test):
        #     _ = self.translate(sample.utterance, plots=['decoder_layer2_block1', 'decoder_layer2_block2'])
        #     print("----------")

        correct_transl = 0
        samples = 0
        for (batch, (t_in, t_out)) in enumerate(val_dataset):
            for i in range(t_in.shape[0]):
                samples += 1
                utt = self.tokenizer_utter.decode([idx for idx in t_in[i, :] if idx < self.tokenizer_utter.vocab_size])
                pred, _ = self.evaluate(utt)  # first try to decode
                min_dim = min(len(t_out[i, :]), len(pred))
                eqls = tf.reduce_all(tf.math.equal(tf.dtypes.cast(pred[:min_dim], tf.int32),
                                                   tf.dtypes.cast(t_out[i, :min_dim], tf.int32)))
                if eqls:
                    correct_transl += 1
                    print(f"Correct: {utt} \n {pred}")
                else:
                    print(f"Wrong: {utt}")
                    pred_utter = self.tokenizer_transl.decode(
                        [idx for idx in pred if idx < self.tokenizer_transl.vocab_size])
                    print(pred_utter)
                    print("-------")
        print(f"Validation accuracy: {(correct_transl / samples):.4f}")

    def translate(self, sentence: str, plots=()) -> str:
        result, attention_weights = self.evaluate(sentence)
        predicted_sentence = self.tokenizer_transl.decode([i for i in result if i < self.tokenizer_transl.vocab_size])
        print('Input: {}'.format(sentence))
        print('Predicted translation: {}'.format(predicted_sentence))
        for plot in plots:
            self.plot_attention_weights(attention_weights, sentence, result, plot)
        return predicted_sentence

    def plot_attention_weights(self, attention, sentence, result, layer):
        fig = plt.figure(figsize=(16, 8))
        sentence = self.tokenizer_utter.encode(sentence)
        attention = tf.squeeze(attention[layer], axis=0)

        for head in range(attention.shape[0]):
            ax = fig.add_subplot(2, 4, head + 1)

            # plot the attention weights
            ax.matshow(attention[head][:-1, :], cmap='viridis')

            fontdict = {'fontsize': 10}
            ax.set_xticks(range(len(sentence) + 2))
            ax.set_yticks(range(len(result)))
            ax.set_ylim(len(result) - 1.5, -0.5)

            ax.set_xticklabels(
                ['<start>'] +
                [self.tokenizer_utter.decode([i]) for i in sentence]
                + ['<end>'],
                fontdict=fontdict, rotation=90)

            ax.set_yticklabels([self.tokenizer_transl.decode([i]) for i in result
                                if i < self.tokenizer_transl.vocab_size],
                               fontdict=fontdict)
            ax.set_xlabel('Head {}'.format(head + 1))
        plt.tight_layout()
        plt.show()

    @classmethod
    def loss_function(cls, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = cls.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    @staticmethod
    def update_accuracy(real, pred, accuracy_object):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        mask = tf.cast(mask, dtype=pred.dtype)
        mask = tf.tile(tf.expand_dims(mask, axis=-1), [1, 1, pred.shape[2]])
        masked_pred = pred * mask
        accuracy_object(real, masked_pred)

    @staticmethod
    def create_masks(inp, tar):
        # Encoder padding mask
        enc_padding_mask = TfUtils.create_padding_mask(inp)
        # Used in the 2nd attention block in the decoder. This padding mask is used to mask the encoder outputs.
        dec_padding_mask = TfUtils.create_padding_mask(inp)
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by the decoder.
        look_ahead_mask = TfUtils.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = TfUtils.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask


if __name__ == "__main__":
    classifier = TransformerTemplateTranslator()
    # classifier.save_dataset()
    classifier.train_model()
    # classifier.evaluate_model()
