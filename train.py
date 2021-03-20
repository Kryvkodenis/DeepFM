import torch
print(torch.__version__)

import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.metrics import Accuracy

from network import DeepFMNet
from data_loader import CustomDataset

EPOCHS = 500
EMBEDDING_SIZE = 5
BATCH_SIZE = 512
NROF_LAYERS = 3
NROF_NEURONS = 50
DEEP_OUTPUT_SIZE = 50
NROF_OUT_CLASSES = 1
LEARNING_RATE = 3e-4
TRAIN_PATH = '/home/denis/repos/sber_risk_DL/week11/data/train_adult.pickle'
VALID_PATH = '/home/denis/repos/sber_risk_DL/week11/data/valid_adult.pickle'

class DeepFM:
    def __init__(self):
        self.train_dataset = CustomDataset(TRAIN_PATH)
        self.train_loader = data_utils.DataLoader(dataset=self.train_dataset,
                                                  batch_size=BATCH_SIZE, shuffle=True)

        self.vall_dataset = CustomDataset(VALID_PATH)
        self.vall_loader = data_utils.DataLoader(dataset=self.vall_dataset,
                                                 batch_size=BATCH_SIZE, shuffle=False)

        self.build_model()

        self.log_params()

        #self.train_writer = SummaryWriter('./logs/train')
        #self.valid_writer = SummaryWriter('./logs/valid')

        return

    def build_model(self):
        self.network = DeepFMNet(nrof_cat=self.train_dataset.nrof_emb_categories, emb_dim=EMBEDDING_SIZE,
                                 emb_columns=self.train_dataset.embedding_columns,
                                 numeric_columns=self.train_dataset.numeric_columns,
                                 nrof_layers=NROF_LAYERS, nrof_neurons=NROF_NEURONS,
                                 output_size=DEEP_OUTPUT_SIZE,
                                 nrof_out_classes=NROF_OUT_CLASSES)

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy()
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)

        return

    def log_params(self):
        return

    def load_model(self, restore_path=''):
        if restore_path == '':
            #self.step = 0
            pass
        else:
            pass

        return

    def run_train(self):
        print('Run train ...')

        self.load_model()

        for epoch in range(EPOCHS):
            self.network.train()

            batch_loss_train = []
            batch_acc_train = []

            for features, label in self.train_loader:
                # Reset gradients
                self.optimizer.zero_grad()

                output = self.network(features)
                # Calculate error and backpropagate
                loss = self.loss(output, label)

                output = torch.sigmoid(output)

                loss.backward()
                acc = self.accuracy(output, label).item()

                # Update weights with gradients
                self.optimizer.step()

                batch_loss_train.append(loss.item())
                batch_acc_train.append(acc)

            #self.train_writer.add_scalar('CrossEntropyLoss', np.mean(batch_loss_train), epoch)
            #self.train_writer.add_scalar('Accuracy', np.mean(batch_acc_train), epoch)

            batch_loss_vall = []
            batch_acc_vall = []

            self.network.eval()
            with torch.no_grad():
                for features, label in self.vall_loader:
                    vall_output = self.network(features)
                    vall_loss = self.loss(vall_output, label)
                    vall_output = torch.sigmoid(vall_output)
                    vall_acc = self.accuracy(vall_output, label).item()
                    batch_loss_vall.append(vall_loss.item())
                    batch_acc_vall.append(vall_acc)

            #self.valid_writer.add_scalar('CrossEntropyLoss', np.mean(batch_loss_vall), epoch)
            #self.valid_writer.add_scalar('Accuracy', np.mean(batch_acc_vall), epoch)

        return


deep_fm = DeepFM()
deep_fm.run_train()