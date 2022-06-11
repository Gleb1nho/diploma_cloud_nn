from cloud_trainer import CloudDetectorLearner
from vizualizer import visualize, show_model_results, show_large_image_results
from cloud_dataset import CloudDataset, ShowCloudDataset

import sys


if __name__ == '__main__':
    help_string = '''
Введите аргументы в следующем порядке:
--train:
    название энкодера
    размер батча для обучения
    директория с набором данных
    количество воркеров для тренировочного загрузчика
    количество воркеров для валидационного загрузчика
    -h или --h для вызова справки
    
    Пример запуска:
main.py --train efficientnet-b0 3 C:/Users/Gleb1nho/Desktop/cv-and-ml/contest/data 4 2

--show:
    директория с набором данных
    путь к сохраненной модели

    Пример запуска:
main.py --show C:/Users/Gleb1nho/Desktop/cv-and-ml/contest/data C:/Users/Gleb1nho/Desktop/diploma/codes/tu-efficientnetv2_s_best_model.pth
    '''

    try:
        if sys.argv[1] == '-h' or sys.argv[1] == '--help':
            print(help_string)
        elif sys.argv[1] == '--train':
            encoder = sys.argv[2]
            batch_size = int(sys.argv[3])

            root_dir = rf'{sys.argv[4]}'
            features_dir = f'{root_dir}/train_features'
            labels_dir = f'{root_dir}/train_labels'

            train_workers = int(sys.argv[5])
            valid_workers = int(sys.argv[6])

            with open('filenames.txt', 'r') as filenames:
                names = list(set(filenames.readlines()))
                train_set = [
                    {
                        'blue': fr'{features_dir}\{i.strip()}\B02.tif',
                        'green': fr'{features_dir}\{i.strip()}\B03.tif',
                        'red': fr'{features_dir}\{i.strip()}\B04.tif',
                        'mask': fr'{labels_dir}\{i.strip()}.tif',
                    }
                    for i in names[0:6000]
                ]
                valid_set = [
                    {
                        'blue': fr'{features_dir}\{i.strip()}\B02.tif',
                        'green': fr'{features_dir}\{i.strip()}\B03.tif',
                        'red': fr'{features_dir}\{i.strip()}\B04.tif',
                        'mask': fr'{labels_dir}\{i.strip()}.tif',
                    }
                    for i in names[6000:7000]
                ]

            train_set = CloudDataset(train_set)
            valid_set = CloudDataset(valid_set)

            trainer = CloudDetectorLearner(
                train_set,
                valid_set,
                train_batch_size=batch_size,
                valid_batch_size=batch_size,
                train_workers_count=train_workers,
                valid_workers_count=valid_workers,
                encoder_name=encoder,
            )

            trainer.start_training()

        elif sys.argv[1] == '--show':
            root_dir = rf'{sys.argv[2]}'
            features_dir = f'{root_dir}/train_features'
            labels_dir = f'{root_dir}/train_labels'

            bst_model = sys.argv[3]

            with open('filenames.txt', 'r') as filenames:
                names = list(set(filenames.readlines()))
                train_set = [
                    {
                        'blue': fr'{features_dir}\{i.strip()}\B02.tif',
                        'green': fr'{features_dir}\{i.strip()}\B03.tif',
                        'red': fr'{features_dir}\{i.strip()}\B04.tif',
                        'mask': fr'{labels_dir}\{i.strip()}.tif',
                    }
                    for i in names[0:6000]
                ]
                valid_set = [
                    {
                        'blue': fr'{features_dir}\{i.strip()}\B02.tif',
                        'green': fr'{features_dir}\{i.strip()}\B03.tif',
                        'red': fr'{features_dir}\{i.strip()}\B04.tif',
                        'mask': fr'{labels_dir}\{i.strip()}.tif',
                    }
                    for i in names[6000:7000]
                ]

            # Пример запуска визуализации входных данных
            # image, mask = train_set[221]
            # visualize(
            #     image=image.permute(1, 2, 0),
            #     cloud_mask=mask.squeeze(),
            # )

            show_model_results(CloudDataset(valid_set), bst_model)
        elif sys.argv[1] == '--large':
            images = [
                '../clouds/large_cloud1_RGB.tif',
                '../clouds/large_cloud2_RGB.tif',
                '../clouds/laege_cloud3_RGB.tif'
            ]
            show_large_image_results(
                ShowCloudDataset(images),
                'C:/Users/Gleb1nho/Desktop/diploma/codes/efficientnet-b0_best_model.pth'
            )
    except IndexError as e:
        print(e)
        print(help_string)
