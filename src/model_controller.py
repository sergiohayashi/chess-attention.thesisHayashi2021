import json
from pathlib import Path

from evaluator import Evaluator
from models import AttentionEncoderDecoderModel
import zipfile
import os
from config import config
from trainer import TrainerController


class ModelPredictController:

    def __init__(self,
                 NUM_LINHAS=2,
                 NO_TEACH=True
                 ):
        self.model = None
        self.NUM_LINHAS = NUM_LINHAS
        self.NO_TEACH = NO_TEACH

    def load(self):
        self.model = AttentionEncoderDecoderModel(NUM_LINHAS=self.NUM_LINHAS, NO_TEACH=self.NO_TEACH).build()

    def useModel(self, model):
        self.model = model
        return self

    def restoreFromCheckpointName(self, trainName):
        checkPointPath = os.path.join(config['CHECKPOINT_FOLDER'], trainName)
        self.model.steps.restoreFromLatestCheckpoint(checkPointPath)

    def restoreFromCheckpointRelativePath(self, relativePath):
        self.model.steps.restoreFromLatestCheckpoint(relativePath)

    def predictOneImage(self, imagePath):
        result = self.model.steps.evaluate(imagePath)
        return result

    def evaluateForTest(self, dataset='test', plot_attention=False, _len=4):
        evaluator = Evaluator(self.model, _len)
        return evaluator.evaluate_test_data(dataset, plot_attention)

    def predictZip(self):
        pass

    def train(self):
        pass

    def save(self):
        pass

    def trainWith(self, level):
        pass

    def resetMetrics(self):
        pass

    def getConfig(self):
        pass

    def getMetrics(self):
        pass


def uncompressToFolder(zipFile, uncompressFolder):
    if os.path.isdir(uncompressFolder):
        print(uncompressFolder, ' already exists. Skip uncompress..')
        return

    print('unzipping ', zipFile, ' to ', uncompressFolder)
    with zipfile.ZipFile(zipFile, 'r') as zip_ref:
        zip_ref.extractall(uncompressFolder)
    print('unzipping ', zipFile, ' to ', uncompressFolder, ' done!')


def save_train_log(trainName, logs):
    logFile = os.path.join(config['LOG_FOLDER'], trainName + ".txt")
    Path(logFile).parents[0].mkdir(parents=True, exist_ok=True)
    with open(logFile, 'w') as file:
        for log in logs:
            file.write(json.dumps(log) + '\n')
    print('arquivo de log gerado em ', logFile)


class ModelTrainController:
    def __init__(self,
                 NUM_LINHAS=2,
                 NO_TEACH=True
                 ):
        self.model = None
        self.trainer = None
        self.NO_TEACH = NO_TEACH
        self.NUM_LINHAS = NUM_LINHAS

    def load(self):
        self.model = AttentionEncoderDecoderModel(NUM_LINHAS=self.NUM_LINHAS, NO_TEACH=self.NO_TEACH).build()

    def useModel(self, model):
        self.model = model

    def initTrainSession(self, BATCH_SIZE=32):
        self.trainer = TrainerController(self.model, BATCH_SIZE=BATCH_SIZE)

    def restoreFromCheckpointName(self, trainName):
        checkPointPath = os.path.join(config['CHECKPOINT_FOLDER'], trainName)
        self.model.steps.restoreFromLatestCheckpoint(checkPointPath)

    def restoreFromCheckpointRelativePath(self, relativePath):
        self.model.steps.restoreFromLatestCheckpoint(relativePath)

    def evaluateForTest(self, dataset='test', _len=4):
        evaluator = Evaluator(self.model, _len)
        return evaluator.evaluate_test_data(dataset)

    def prepareDatasetForTrain(self, datasetZipFileOrFolder, use_sample=(0.1, 0.1)):
        if datasetZipFileOrFolder.endswith('.zip'):
            print('preparing dataset from zip file ', datasetZipFileOrFolder)
            uncompressFolder = os.path.join(config['TMP_TRAIN_FOLDER'],
                                            os.path.basename(datasetZipFileOrFolder).replace('.zip', ''))
            uncompressToFolder(datasetZipFileOrFolder, uncompressFolder)
        else:
            uncompressFolder = datasetZipFileOrFolder

        # prepare dataset
        self.trainer.prepareFilesForTrain(uncompressFolder, use_sample)
        print('Dataset from zip file ', datasetZipFileOrFolder, ' ready for training')

    def trainUntil(self, target_loss, target_acc, min_max_epoch, lens=[4], train_name='none', test_set=None):
        print('starting trainUntil ', target_loss, min_max_epoch, train_name)
        self.trainer.trainUntil(target_loss, target_acc, min_max_epoch, lens, train_name, test_set=test_set)
        print('starting trainUntil ', target_loss, min_max_epoch, train_name, ' DONE!')

    def save(self, trainName):
        checkPointPath = os.path.join(config['CHECKPOINT_FOLDER'], trainName)
        self.model.steps.saveCheckpointTo(checkPointPath)
        print('model saved to ' + checkPointPath)

    def levelCheckpointExists(self, trainName):
        checkPointPath = os.path.join(config['CHECKPOINT_FOLDER'], trainName)
        return self.model.steps.checkpointExists(checkPointPath)

    def trainOrContinueForCurriculum(self, curriculumName, levelsDatasetZipFiles,
                                     target_loss, target_acc, min_max_epoch, use_sample=(0.1, 0.1), lens=[4],
                                     test_set=None):

        if self.levelCheckpointExists(curriculumName):
            print('treino já finalizado. Checkpoint em ', self.levelCheckpointExists(curriculumName));
            return

        skip = True
        for levelZipFile in levelsDatasetZipFiles:
            checkpointName = "{}--{}".format(curriculumName, os.path.basename(levelZipFile).replace('.zip', ''))

            if skip and self.levelCheckpointExists(checkpointName):
                # se treino ja foi finalizado, somente recupera..
                print('treino para ', checkpointName, ' ja realizado. Recupera checkpoint')
                ModelPredictController().useModel(self.model).restoreFromCheckpointName(checkpointName)
            else:
                # caso contrario, faz o treinamento
                print('treino para ', checkpointName, ' NAO realizado. Realiza treinamento.')
                self.prepareDatasetForTrain(levelZipFile, use_sample)
                self.trainUntil(target_loss, target_acc, min_max_epoch, lens, checkpointName, test_set)
                self.save(checkpointName)
                skip = False

                # valida no testset, até o tamanho maximo informado
                # evaluator = Evaluator(self.model, target_len=lens[-1])
                # test_acc = evaluator.evaluate_test_data('teste' if self.NUM_LINHAS == 2 else 'test-8lines')

                # salva em aquivo
                save_train_log(checkpointName, self.trainer.logs)

        self.save(curriculumName)
        print('treinamento de curriculo finalizado com sucesso!')
