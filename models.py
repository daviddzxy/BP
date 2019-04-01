import torch
import numpy as np


class Model():
    def __init__(self, epoch_count, learning_rate, network, loss_function):
        self.epoch_count = epoch_count
        self.learning_rate = learning_rate
        self.network = network
        self.loss_function = loss_function

    def fit(self, dataloader):
        pass

    def predict(self, dataloader):
        pass


class SiameseModel(Model):
    def __init__(self, epoch_count, learning_rate, network, loss_function):
        super(SiameseModel, self).__init__(epoch_count, learning_rate, network, loss_function)

    def fit(self, dataloader):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        loss_history = []
        for epoch in range(0, self.epoch_count):
            for i, data in enumerate(dataloader, 0):
                image0, image1, pair_label, label0, label1 = data
                pair_label = pair_label.type(torch.FloatTensor)
                image0, image1, pair_label = image0.cuda(), image1.cuda(), pair_label.cuda()
                out1, out2 = self.network(image0, image1)
                loss_contrastive = self.loss_function(out1, out2, pair_label)
                loss_contrastive.backward()
                optimizer.step()

                print("Epoch no. {}\nBatch {}\nCurrent loss {}\n".format(epoch, i, loss_contrastive.item()))
                loss_history.append(loss_contrastive.item())

        return loss_history

    def predict(self, dataloader):
        encodings = []
        labels = []
        with torch.no_grad():
            self.network.eval()
            for img, label in dataloader:
                img = img.cuda()
                out = self.network.get_encoding(img)
                out = out.cpu()
                out = out.numpy()
                label = label.numpy()
                encodings.extend(out)
                labels.extend(label)

        return np.array(encodings), np.array(labels)
