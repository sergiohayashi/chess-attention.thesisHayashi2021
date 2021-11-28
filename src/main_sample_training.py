from model_controller import ModelTrainController


def simple_train():
    lens = [16]
    niveis = [
        '../dataset/-8linhas-handwritten-only-1000.zip',
    ]
    train_name = "simple-8lines-training"

    model = ModelTrainController(NUM_LINHAS=8, NO_TEACH=False)
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.25, 0.90,
                                       (1, 1),   # 3 epochs, para teste somente...
                                       (1000, 200),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)


if __name__ == '__main__':
    simple_train()
