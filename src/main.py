from glob import glob

from model_controller import ModelPredictController, ModelTrainController


def predict():
    model = ModelPredictController();
    model.load()
    # model.model.print_summary()
    model.restoreFromBestCheckpoint()
    model.evaluateForTest()
    # model.model.print_summary()
    print('predicted sample-1=> ', model.predictOneImage("./sample_data/sample-1.jpg"))
    print('predicted sample-2=> ', model.predictOneImage("./sample_data/sample-2.jpg"))


def predictFrom(trainName, n=1):
    model = ModelPredictController();
    model.load()
    model.restoreFromBestCheckpoint()
    print('------------- best checkpoint ------------------')
    model.evaluateForTest()
    model.restoreFromCheckpointName(trainName)
    print('------------- ', trainName, '------------------')
    for _ in range(0, n):
        model.evaluateForTest()


def train(trainName):
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.prepareDatasetForTrain("../dataset/nivel-0--dataset-v034--2lines-parts--42k.zip")


def train_curriculum():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-1', [
        '../dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 1.0, 2)


def train_curriculum_2():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-2', [
        '../dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 1.0, 2)


def train_curriculum_3():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-3', [
        '../dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 1.0, 2)


def train_curriculum_4():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-4', [
        '../dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 1.0, 2)


def train_mixed_try11():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-mixed-try11', [
        '../dataset/mixed.zip'
    ], 0.01, 50, (25000, 500))

    '''
        Resultado:
        len 1 accuracy 0.8157894611358643 cir 0.102339186
        len 2 accuracy 0.8070175647735596 cir 0.1125731
        len 3 accuracy 0.8157894611358643 cir 0.10550683
        len 4 accuracy 0.8004385828971863 cir 0.11878655
    '''


def train_curriculum_8():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-8', [
        '../dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.1, 50, (25000, 500))


def train_curriculum_10():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-10', [
        '../dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.01, 30, (25000, 500))


def train_curriculum_13():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-13', [
        '../dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.05, 0.98, 30, (25000, 500), lens=[1, 2, 3, 4])


def train_1003_5__1006_():
    model = ModelTrainController();
    model.load()
    model.restoreFromCheckpointName('from-1003-5')
    model.evaluateForTest()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('_1006_', [
        '../dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.07, 0.98, 50, (0, 0), lens=[1, 2, 3, 4])


def train_curriculum_14():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-14', [
        '../dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.07, 0.97, (2, 30), (0, 0), lens=[1, 2, 3, 4])


def train_from_1003_():
    model = ModelTrainController();
    model.load()
    model.restoreFromCheckpointRelativePath('../other_checkpoints/1003/checkpoints/train')
    model.evaluateForTest()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('local-1005-1-', [
        '../dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
    ], 0.1, 0.98, (2, 50), (0.8, 0.8), lens=[1, 2, 3, 4])


def train_from_1005_1_():  # result: 0.86	0.81	0.83	0.79
    model = ModelTrainController();
    model.load()
    model.restoreFromCheckpointName('local-1005-1-')
    model.evaluateForTest()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('local-1006-1-', [
        '../dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.1, 0.98, (2, 50), (0.8, 0.8), lens=[1, 2, 3, 4])


def level7_try1_():
    model = ModelTrainController();
    model.load()
    model.restoreFromCheckpointRelativePath('../other_checkpoints/1006/checkpoints/train')
    model.evaluateForTest()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('level7_try1_', [
        '../dataset/mixed.zip'
    ], 0.025, 0.99, (2, 50), (1, 1), lens=[1, 2, 3, 4])


def evaluate_best():
    model = ModelPredictController()
    model.load()
    model.restoreFromBestCheckpoint()
    model.evaluateForTest()
    model.evaluateForTest('irt_hebraica_jan2020')


def train_curriculum_21():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-21', [
        '../dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        # , '../dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.07, 0.97, (2, 20), (1, 1), lens=[1, 2, 3, 4])


def best_add_hebraica():
    model = ModelTrainController();
    model.load()
    model.restoreFromCheckpointRelativePath('../other_checkpoints/1006/checkpoints/train')
    model.evaluateForTest()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('nivel-6', [
        '../train-folder/tmp/nivel-6--hebraica-metade-1'
    ], 0.025, 0.99, (1, 50), (1, 1), lens=[1, 2, 3, 4])


def train_handwritten_only():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('handwritten-only', [
        '../dataset/handwritten-only.zip'
    ], 0.1, 0.95, (2, 50), (1, 1), lens=[1, 2, 3, 4])


def train_handwritten_only_deumavez():
    model = ModelTrainController();
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('ca', [
        '../dataset/handwritten-only.zip'
    ], 0.1, 0.95, (2, 50), (1, 1), lens=[4])


def train_handwritten_only_deumavez_teach_forcing():
    model = ModelTrainController(NO_TEACH=False);
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('handwritten-only-deumavez--teach-force-final', [
        '../dataset/handwritten-only.zip'
    ], 0.1, 0.95, (2, 50), (1, 1), lens=[1, 2, 3, 4])


def train_curriculum_21_teacher_force():
    model = ModelTrainController(NO_TEACH=False);
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum('curriculum-try-22--teacher-force', [
        '../dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ], 0.07, 0.97, (2, 20), (1, 1), lens=[1, 2, 3, 4])


def train_all_combinations():
    dataset_handwritten = ['../dataset/handwritten-only.zip']

    dataset_niveis = [
        '../dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ]

    for lens in [[4], [1, 2, 3, 4]]:
        for no_teach in [True, False]:
            for niveis in [dataset_handwritten, dataset_niveis]:
                train_name = "train_200210902_{}__{}__{}".format(
                    "FIX_LEN" if len(lens) <= 1 else "INCR_LEN",
                    "NO_TEACH" if no_teach else "TEACH",
                    "CURRICULUM" if len(niveis) > 1 else "HANDWRITTEN_ONLY")

                print("========================================================================")
                print("INICIANDO TESTE => ", train_name)

                model = ModelTrainController(NO_TEACH=no_teach)
                model.load()
                model.initTrainSession()
                model.trainOrContinueForCurriculum(train_name,
                                                   niveis, 0.1, 0.95,
                                                   (1, 50),
                                                   (25000, 5000),
                                                   lens=lens)  # testa 2 epoch por enquanto..
                #
                print("VALIDANDO => ", train_name)
                # # model.evaluateForTest('test')
                model.evaluateForTest('irt_hebraica_jan2020')
                print("FINALIZADO TESTE => ", train_name)


def train_all_combinations__um_a_um():
    dataset_niveis = [
        '../dataset/nivel-0--dataset-v034--2lines-parts--42k.zip'
        , '../dataset/nivel-1--dataset-v034--2lines-parts--42k-v3.zip'
        , '../dataset/nivel-2--dataset-v034--2lines-parts--40k-v4.zip'
        , '../dataset/nivel-4--dataset-v035--2lines-32k-v5.0.zip'
        , '../dataset/nivel-5--dataset-v035--2lines-32k-v5.1.zip'
    ]

    ## Ajustar 1 a 1
    lens = [4]  # [[4], [1, 2, 3, 4]]:
    no_teach = True  # [True, False]
    niveis = dataset_niveis  # [dataset_handwritten, dataset_niveis]

    train_name = "train_200210902_1a1_{}__{}__{}".format(
        "FIX_LEN" if len(lens) <= 1 else "INCR_LEN",
        "NO_TEACH" if no_teach else "TEACH",
        "CURRICULUM" if len(niveis) > 1 else "HANDWRITTEN_ONLY")

    print("========================================================================")
    print("INICIANDO TESTE => ", train_name)

    model = ModelTrainController(NO_TEACH=no_teach)
    model.load()
    model.initTrainSession()
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.1, 0.95,
                                       (1, 50),
                                       (25000, 5000),
                                       lens=lens)  # testa 2 epoch por enquanto..
    #
    print("VALIDANDO => ", train_name)
    model.evaluateForTest('irt_hebraica_jan2020')
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_handwritten():
    lens = [2, 4, 6, 8, 10, 12, 14, 16]
    no_teach = True
    niveis = ['../dataset/dataset-8lines-v002-somente_handwriten.zip']
    NUM_LINES = 8

    train_name = "train_20211026_final_{}lines_{}__{}__{}".format(
        NUM_LINES,
        "FIX_LEN" if len(lens) <= 1 else "INCR_LEN",
        "NO_TEACH" if no_teach else "TEACH",
        "CURRICULUM" if len(niveis) > 1 else "HANDWRITTEN_ONLY")

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=no_teach)
    model.load()
    # model.initTrainSession()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.1, 0.90,
                                       (1, 50),
                                       (25000, 5000),
                                       lens=lens)
    model.evaluateForTest('test-8lines', len=lens[-1])
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_curriculum():
    #    lens = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16]   # etapas 1
    #    lens = [1, 2, 4, 6, 8, 10, 12, 14, 16]   # etapas 2, 3
    lens = [16]  # etapas 4, 5 DIRETO
    lens = [2, 4, 6, 8, 10, 12, 14, 16]  # etapas 4
    no_teach = True
    niveis = [
        '../dataset/curriculum-8-linhas--etapa-1.zip',
        '../dataset/curriculum-8-linhas--etapa-2.zip',
        '../dataset/curriculum-8-linhas--etapa-3.zip',
        '../dataset/curriculum-8-linhas--etapa-4.zip',
        '../dataset/curriculum-8-linhas--etapa-5.zip',
    ]
    NUM_LINES = 8

    train_name = "train_20211026_curriculum_try2_{}lines_{}__{}__{}".format(
        NUM_LINES,
        "INCR_LEN",
        "NO_TEACH" if no_teach else "TEACH",
        "CURRICULUM" if len(niveis) > 1 else "HANDWRITTEN_ONLY")

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=no_teach)
    model.load()
    # model.initTrainSession()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.1, 0.90,
                                       (1, 50),
                                       (10000, 1000),
                                       lens=lens)
    model.evaluateForTest('test-8lines', _len=16)
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_curriculum_etapa5():
    lens = [2, 4, 6, 8, 10, 12, 14, 15, 16]  # etapas 5
    no_teach = True
    niveis = [
        '../dataset/curriculum-8-linhas--etapa-1.zip',
        '../dataset/curriculum-8-linhas--etapa-2.zip',
        '../dataset/curriculum-8-linhas--etapa-3.zip',
        '../dataset/curriculum-8-linhas--etapa-5.zip',
    ]
    NUM_LINES = 8

    train_name = "train_20211026_curriculum_try2_{}lines_{}__{}__{}".format(
        NUM_LINES,
        "INCR_LEN",
        "NO_TEACH" if no_teach else "TEACH",
        "CURRICULUM" if len(niveis) > 1 else "HANDWRITTEN_ONLY")

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=no_teach)
    model.load()
    # model.initTrainSession()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       # niveis, 0.1, 0.90,
                                       niveis, 0.1, 0.9,
                                       (1, 50),
                                       (20000, 1000),
                                       lens=lens)
    model.evaluateForTest('test-8lines', _len=16)
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_curriculum_etapa5_refinamento_1():
    lens = [16]  # etapas 5
    no_teach = True
    niveis = [
        '../dataset/curriculum-8-linhas--etapa-5.zip',
    ]
    NUM_LINES = 8

    train_name = "train_20211026_curriculum_try2_8lines_INCR_LEN__NO_TEACH__CURRICULUM_ref1_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=no_teach)
    model.load()
    model.restoreFromCheckpointName(
        'train_20211026_curriculum_try2_8lines_INCR_LEN__NO_TEACH__CURRICULUM--curriculum-8-linhas--etapa-5')
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       # niveis, 0.1, 0.90,
                                       niveis, 0.1, 0.99,
                                       (1, 50),
                                       (20000, 1000),
                                       lens=lens)
    model.evaluateForTest('test-8lines', _len=16)
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_curriculum_etapa5__direto():
    lens = [2]  # etapas 5
    no_teach = True
    niveis = [
        '../dataset/curriculum-8-linhas--etapa-5.zip',
    ]
    NUM_LINES = 8

    train_name = "train_20211026_curriculum_etapa5_direto_8lines_INCR_LEN__NO_TEACH__CURRICULUM_--testfly-6"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=no_teach)
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.25, 0.75,
                                       # niveis, 0.1, 0.99,
                                       # (1, 50),
                                       (1, 2),
                                       # (20000, 1000),
                                       (1000, 100),
                                       lens=lens)
    model.evaluateForTest('test-8lines', _len=16)
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_comparativo_20211106_curriculum():
    lens = [16]
    no_teach = True
    niveis = [
        '../dataset/curriculum-8-linhas--etapa-1.zip',
        '../dataset/curriculum-8-linhas--etapa-2.zip',
        '../dataset/curriculum-8-linhas--etapa-3.zip',
        '../dataset/curriculum-8-linhas--etapa-4.zip',
        '../dataset/curriculum-8-linhas--etapa-5.zip',
    ]
    NUM_LINES = 8

    train_name = "train_comparativo_20211106_curriculum_4_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=no_teach)
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.25, 0.90,
                                       (1, 100),
                                       (5000, 500),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)
    # acc_test_16ln => 0.5137061476707458


def train_8_lines_comparativo_20211106_curriculum_unico():
    lens = [16]
    no_teach = True
    niveis = [
        '../dataset/-8linhas-curriculum--unico--5k.zip',
    ]
    NUM_LINES = 8

    train_name = "train_comparativo_20211106_curriculum_unico_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=no_teach)
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.25, 0.90,
                                       (1, 500),
                                       (5000, 500),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)
    # acc_test_16ln => 0.4150219261646271


def train_8_lines_comparativo_20211106_handwritten():
    lens = [16]
    no_teach = True
    niveis = [
        '../dataset/-8linhas-handwritten--5k.zip',
    ]
    NUM_LINES = 8

    train_name = "train_comparativo_20211106_handwritten_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=no_teach)
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.25, 0.90,
                                       (1, 500),
                                       (5000, 500),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)
    # acc_test_16ln => 0.5137061476707458


def train_8_lines_comparativo_20211106_handwritten_only_2388():
    # lens = [1, 2, 4, 8, 12, 15, 16]
    lens = [16]
    no_teach = True
    niveis = [
        '../dataset/-8linhas-handwritten-only-2388.zip',
    ]
    NUM_LINES = 8

    train_name = "train_comparativo_20211106_handwritten_only_2388_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=no_teach)
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.25, 0.90,
                                       (1, 500),
                                       (2000, 388),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    # model.evaluateForTest('test-8lines', _len=16)
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_comparativo_20211106_random_only_2388():
    # lens = [1, 2, 4, 8, 12, 15, 16]
    lens = [16]
    no_teach = True
    niveis = [
        '../dataset/-8linhas-random-only-2388.zip',
    ]
    NUM_LINES = 8

    train_name = "train_comparativo_20211106_random_only_2388_try3_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=no_teach)
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.25, 0.90,
                                       (1, 500),
                                       (2000, 388),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_comparativo_20211106_random_only_2388_incr():
    lens = [1, 2, 4, 6, 8, 10, 12, 14, 15, 16]
    no_teach = True
    niveis = [
        '../dataset/-8linhas-random-only-2388.zip',
    ]
    NUM_LINES = 8

    train_name = "train_comparativo_20211106_random_only_2388_incr_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=no_teach)
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.25, 0.90,
                                       (1, 500),
                                       (2000, 388),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_comparativo_20211106_handwritten_teacher():
    lens = [16]
    niveis = [
        '../dataset/-8linhas-handwritten--5k.zip',
    ]
    NUM_LINES = 8

    train_name = "train_comparativo_20211106_handwritten_teacher_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.25, 0.90,
                                       (1, 500),
                                       (5000, 500),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_comparativo_20211106_handwritten_teacher__2():
    # lens = [1, 2, 4, 8, 12, 15, 16]
    lens = [16]
    niveis = [
        '../dataset/-8linhas-handwritten--5k.zip',
    ]
    NUM_LINES = 8

    train_name = "train_comparativo_20211106_handwritten_teacher_2_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()
    model.restoreFromCheckpointName(
        'train_comparativo_20211106_handwritten_teacher_---8linhas-handwritten--5k')
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.1, 0.99,
                                       (1, 500),
                                       (5000, 500),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_comparativo_20211106_curriculum_etapa6():
    # lens = [1, 2, 4, 8, 12, 15, 16]
    lens = [16]
    no_teach = True
    niveis = [
        '../dataset/curriculum-8-linhas--etapa-6-5500.zip',
    ]
    NUM_LINES = 8

    train_name = "train_comparativo_20211106_curriculum_4_etapa_6_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=no_teach)
    model.load()
    model.restoreFromCheckpointName(
        'train_comparativo_20211106_curriculum_4_--curriculum-8-linhas--etapa-5')
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.25, 0.90,
                                       (1, 100),
                                       (5000, 500),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)
    # acc_test_16ln => 0.5137061476707458


def train_8_lines_comparativo_20211106_random_1500parts_():
    lens = [16]
    no_teach = True
    niveis = [
        '../dataset/dataset-v022-7k-shuffle-parts1500.zip',
    ]
    NUM_LINES = 8

    train_name = "train_comparativo_20211106_random_parts1500_2388_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=no_teach)
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.25, 0.90,
                                       (1, 500),
                                       (2000, 388),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_comparativo_20211106_random_1500parts_1000_():
    lens = [16]
    no_teach = True
    niveis = [
        '../dataset/dataset-v022-7k-shuffle-parts1500.zip',
    ]
    NUM_LINES = 8

    train_name = "train_comparativo_20211106_random_parts1500_1000_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=no_teach)
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.25, 0.90,
                                       (1, 500),
                                       (1000, 200),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_comparativo_20211106_handwritten_teacher_10k_():
    lens = [16]
    niveis = [
        '../dataset/-8linhas-handwritten--10k.zip',
    ]
    NUM_LINES = 8

    train_name = "train_comparativo_20211106_handwritten_teacher_10k_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()
    model.restoreFromCheckpointName(
        'train_comparativo_20211106_handwritten_teacher_---8linhas-handwritten--5k')
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.07, 0.97,
                                       (1, 500),
                                       (10000, 1000),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_comparativo_20211106_handwritten_5k_2_():
    # lens = [1, 2, 4, 8, 12, 15, 16]
    lens = [16]
    niveis = [
        '../dataset/-8linhas-handwritten--5k.zip',
    ]
    NUM_LINES = 8

    train_name = "train_comparativo_20211106_handwritten_5k_cont-1_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=True)
    model.load()
    model.restoreFromCheckpointName(
        'train_comparativo_20211106_handwritten_---8linhas-handwritten--5k')
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.1, 0.95,
                                       (1, 500),
                                       (5000, 500),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_8_lines_comparativo_20211106_handwritten_teacher_tam_menor():
    lens = [16]
    niveis = [
        '../dataset/-8linhas-handwritten--5k.zip',
    ]
    NUM_LINES = 8

    #
    # Este teste é rodado com tamanho menor informado em config.py!!!
    #
    train_name = "train_comparativo_20211106_handwritten_teacher_tam_menor"

    from config import config
    config["FORCE_INPUT_SIZE"] = {"INPUT_SHAPE": (400, 430), "ATTENTION_SHAPE": (25, 26)}
    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.25, 0.90,
                                       (1, 500),
                                       (5000, 500),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )
    # model.evaluateForTest('tes t-8lines', _len=16)
    print("FINALIZADO TESTE => ", train_name)


def train_8_lines_comparativo_20211106_handwritten_tam_menor():
    lens = [16]
    niveis = [
        '../dataset/-8linhas-handwritten--5k.zip',
    ]
    NUM_LINES = 8

    from config import config
    config["FORCE_INPUT_SIZE"] = {"INPUT_SHAPE": (400, 430), "ATTENTION_SHAPE": (25, 26)}
    train_name = "train_comparativo_20211106_handwritten_tam_menor_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=True)
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.25, 0.90,
                                       (1, 500),
                                       (5000, 500),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_8_lines_comparativo_20211106_random_tam_menor():
    lens = [16]
    niveis = [
        '../dataset/curriculum-8-linhas--etapa-1.zip',
    ]
    NUM_LINES = 8

    from config import config
    config["FORCE_INPUT_SIZE"] = {"INPUT_SHAPE": (400, 430), "ATTENTION_SHAPE": (25, 26)}
    train_name = "train_comparativo_20211106_random_tam_menor_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=True)
    model.load()
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.25, 0.90,
                                       (1, 500),
                                       (5000, 500),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_8_lines_comparativo_20211106_handwritten_teacher_10k_cont_2_():
    lens = [16]
    niveis = [
        '../dataset/-8linhas-handwritten--10k.zip',
    ]
    NUM_LINES = 8

    train_name = "train_comparativo_20211106_handwritten_teacher_10k_3_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()
    model.restoreFromCheckpointName(
        'train_comparativo_20211106_handwritten_teacher_10k_')
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.01, 0.99,
                                       (1, 100),
                                       (10000, 1000),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


def train_8_lines_comparativo_20211106_handwritten_teacher_10k_cont_hebraica_():
    lens = [16]
    niveis = [
        '../dataset/-8linhas-handwritten--5k--with-hebraica-.zip',
    ]
    NUM_LINES = 8

    train_name = "train_comparativo_20211106_handwritten_teacher_10k_ref-hebraica_"

    model = ModelTrainController(NUM_LINHAS=NUM_LINES, NO_TEACH=False)
    model.load()
    model.restoreFromCheckpointName(
        'train_comparativo_20211106_handwritten_teacher_10k_')
    model.initTrainSession(BATCH_SIZE=16)
    model.trainOrContinueForCurriculum(train_name,
                                       niveis, 0.07, 0.97,
                                       (1, 100),
                                       (5109, 500),
                                       lens=lens,
                                       test_set='test-8lines'
                                       )


if __name__ == '__main__':
    train_8_lines_comparativo_20211106_handwritten_teacher_10k_cont_hebraica_()
