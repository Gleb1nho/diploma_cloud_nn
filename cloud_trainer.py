import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader


class CloudDetectorLearner:
    def __init__(
            self,
            train_set,
            valid_set,
            encoder_name='efficientnet-b0',
            encoder_weights='imagenet',
            activation='sigmoid',
            device='cuda',
            epochs_count=50,
            train_batch_size=1,
            valid_batch_size=1,
            train_workers_count=1,
            valid_workers_count=1,
    ):
        self._encoder_name = encoder_name
        self._encoder_weights = encoder_weights
        self._activation = activation
        self._device = device
        self._epochs_count = epochs_count

        self._model = smp.Unet(
            encoder_name=self._encoder_name,
            encoder_weights=self._encoder_weights,
            in_channels=3,
            classes=1,
            activation=self._activation
        )

        self._train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=train_workers_count
        )

        self._valid_loader = DataLoader(
            valid_set,
            batch_size=valid_batch_size,
            shuffle=False,
            num_workers=valid_workers_count
        )

        self._loss = smp.utils.losses.JaccardLoss()
        self._metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
        ]
        self._optimizer = torch.optim.Adam([
            dict(params=self._model.parameters(), lr=1e-4),
        ])

        self._train_epoch = smp.utils.train.TrainEpoch(
            self._model,
            loss=self._loss,
            metrics=self._metrics,
            optimizer=self._optimizer,
            device=self._device,
            verbose=True,
        )

        self._valid_epoch = smp.utils.train.ValidEpoch(
            self._model,
            loss=self._loss,
            metrics=self._metrics,
            device=self._device,
            verbose=True,
        )

        self._max_score = 0

    def start_training(self):
        print(f'Запуск обучения модели с энкодером {self._encoder_name}')
        logs_path = f'../logs/{datetime.date(datetime.now())}{self._encoder_name}_unet_loss_jaccard'

        for i in range(0, self._epochs_count):
            print('\nEpoch: {}'.format(i))

            train_logs = self._train_epoch.run(self._train_loader)

            valid_logs = self._valid_epoch.run(self._valid_loader)

            writer = SummaryWriter(logs_path)
            writer.add_scalar('Accuracy/train', train_logs['iou_score'], i)
            writer.add_scalar('Accuracy/valid', valid_logs['iou_score'], i)
            writer.add_scalar('Loss/train', train_logs['jaccard_loss'], i)
            writer.add_scalar('Loss/valid', valid_logs['jaccard_loss'], i)
            writer.close()

            # do something (save model, change lr, etc.)
            if self._max_score < valid_logs['iou_score']:
                self._max_score = valid_logs['iou_score']
                torch.save(self._model, '../best_model.pth')
                print('Model saved!')

            if i == 25:
                self._optimizer.param_groups[0]['lr'] = 1e-6
                print('Decrease decoder learning rate to 1e-6!')
