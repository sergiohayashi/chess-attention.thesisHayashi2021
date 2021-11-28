import os
from _ast import expr
from glob import glob
from pathlib import Path

from plotter import Plotter
from utils import read_label
import tensorflow as tf
import nltk
from nltk.metrics import distance


def cir_word(hp, gt):
    return distance.edit_distance(hp, gt) / len(gt)


def cir_line(predicted, expected):
    return tf.reduce_mean([cir_word(p, e) for (p, e) in zip(predicted, expected)]).numpy()


def cir_set(result, labels, _len=None):
    if _len is None:
        _len = len(labels)
    return tf.reduce_mean([cir_line(p[:_len], e[:_len]) for (p, e) in zip(result, labels)]).numpy()


class Evaluator:
    def __init__(self, model, target_len=4):
        print('target_len= ', target_len)
        self._len = target_len
        self.model = model
        self.plotter = Plotter(model)

    @staticmethod
    def load_from_path(path, max=None):
        test_images = glob(os.path.join(path, 'images/*.jpg'))
        test_images.sort()

        test_labels_files = glob(os.path.join(path, 'labels/*.pgn'))
        test_labels_files.sort()
        test_labels = [read_label(f) for f in test_labels_files]
        # test_labels= [cleanup( x).lower() for x in test_labels]
        test_labels = [label.split() for label in test_labels]
        if max is None:
            return test_images, test_labels
        else:
            return test_images[:max], test_labels[:max]

    @staticmethod
    def load_test(dataset):
        return Evaluator.load_from_path('../data/test-data/' + dataset, max=None)

    def evaluate_test_data(self, dataset, plot_attention=False):
        result_acc = []
        print('evaluating dataset ', dataset)
        ac, cer, predicted, expected = self.evaluate_all_data(*Evaluator.load_test(dataset), self._len,
                                                              plot_attention=plot_attention)
        result_acc.append((ac, cer, dataset))
        return result_acc

    def evaluate_all_data(self, images, labels, maxlen, plot_attention=False):
        result = []
        all_results = []
        print('evaluating total images: ', len(images), '...')

        for i in range(0, len(images)):
            if i % 100 == 0:
                print('evaluating ', i, '...')
            r, attention_plot, _ = self.model.steps.evaluate(images[i], maxlen)
            result.append(r)

            # calcula o indice
            m = tf.keras.metrics.Accuracy()
            m.update_state(
                self.model.tokenizer.texts_to_sequences(labels[i])[:maxlen],
                self.model.tokenizer.texts_to_sequences(r)[:maxlen])
            all_results.append({
                'file': images[i],
                'label': labels[i],
                'prediction': r,
                'attention': attention_plot,
                'acc': float(m.result()),
                'cer': cir_set([r], [labels[i]], maxlen)
            });

        # calcula a acurácia para cada tamanho
        # print('--------------< Indice por tamanho de sequencia >----------------------------')
        result_ac = []
        result_cer = []
        for _len in range(1, maxlen + 1):
            m = tf.keras.metrics.Accuracy()

            # acurácia para cada teste, até o tamanho atual
            for i in range(0, len(result)):
                useLen = min(len(labels[i]), len(result[i]), _len)
                m.update_state(
                    self.model.tokenizer.texts_to_sequences(labels[i])[:useLen],
                    self.model.tokenizer.texts_to_sequences(result[i])[:useLen])

            # print('len', _len, 'accuracy', float(m.result()), 'cir', cir_set(labels, result, _len))
            result_ac.append(float(m.result()))
            result_cer.append(float(cir_set(result, labels, _len)))

        if plot_attention:
            # ordena, da pior para melhor
            # all_results = sorted(all_results, key=lambda k: (k['acc'], -k['cer']))
            all_results = sorted(all_results, key=lambda k: (-k['cer'], k['acc']))

            # imprime
            for i in range(0, len(all_results)):
                r = all_results[i]
                print('--------------------< ', i, ': ', Path(r['file']).name, '>------------------------------')
                print('len:', maxlen, 'acc:', r['acc'], 'cer', r['cer'], 'file: ', r['file'])
                self.plotter.plot_attention(r['file'], r['prediction'], r['attention'], r['label'])

        predicted = result
        return result_ac, result_cer, predicted, labels
