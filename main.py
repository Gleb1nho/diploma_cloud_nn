from cloud_trainer import CloudDetectorLearner
from vizualizer import visualize, show_model_results
from cloud_dataset import CloudDataset


if __name__ == '__main__':

    root_dir = r'C:\Users\Gleb1nho\Desktop\cv-and-ml\contest\data'
    features_dir = r'C:\Users\Gleb1nho\Desktop\cv-and-ml\contest\data\train_features'
    labels_dir = r'C:\Users\Gleb1nho\Desktop\cv-and-ml\contest\data\train_labels'

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

    # Пример запуска визуализации входных данных
    # image, mask = train_set[221]
    # visualize(
    #     image=image.permute(1, 2, 0),
    #     cloud_mask=mask.squeeze(),
    # )

    trainer = CloudDetectorLearner(
        train_set,
        valid_set,
        train_batch_size=2,
        valid_batch_size=1,
        train_workers_count=4,
        valid_workers_count=2,
        encoder_name='tu-efficientnetv2_m'
    )

    trainer.start_training()

    # show_model_results(valid_set, '../efficientnet-b1_best_model.pth')
