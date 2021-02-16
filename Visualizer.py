import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def __init__(self, list_global_sparsity, list_epochs, lists_training_loss, list_test_loss):
        super().__init__()
        self.list_global_sparsity = np.array(list_global_sparsity)  # list of all pruning amounts applied
        self.list_epochs = np.array(list_epochs)+1  # list of all epochs computed
        self.lists_training_loss = np.array(lists_training_loss)  # list of list of losses
        self.list_test_loss = np.array(list_test_loss)

        for loss_list in self.lists_training_loss:
            if len(loss_list) > len(self.list_epochs):
                print("Loss lists length does not match the number of epochs (list_epochs)")

        if len(self.list_global_sparsity) != len(self.lists_training_loss):
            print("Losses missing in lists_training_loss (does not match the number of pruning amount applied)")

    def show_training_loss_pruning(self, indexes=None):
        """Plots the loss's evolution for each pruned version of the model
            Loss function corresponding to indexes (a,b) (b excluded)"""
        if indexes is None:
            a, b = 0, len(self.list_global_sparsity)
        else:
            a, b = indexes

        f, ax = plt.subplots(figsize=(10, 10))
        for i in range(a, b):
            list_loss = self.lists_training_loss[i]
            ax.plot(self.list_epochs[:len(list_loss)], list_loss/np.mean(self.lists_training_loss[0]), label="Sp = {:.2f}".format(self.list_global_sparsity[i]))
        ax.set_title("Training loss on MNIST")
        ax.set_xlabel("number of epochs")
        ax.set_ylabel("Training loss")
        ax.legend()
        return ax

    def show_score_pruning(self):
        """Plots the evolution of the last training when the pruning ratio increase"""
        list_final_loss = np.array([elt[-1] for elt in self.lists_training_loss])
        f, ax = plt.subplots(figsize=(10, 10))
        ax.plot(self.list_global_sparsity, list_final_loss/list_final_loss[0], label="loss")
        ax.set_title("Training loss after {} epochs".format(self.list_epochs[-1]))
        ax.set_xlabel("Global Sparsity")
        ax.set_ylabel("Training loss")
        ax.legend()
        return ax

    def show_test_loss(self):
        f, ax = plt.subplots(figsize=(10, 10))
        ax.plot(self.list_global_sparsity, self.list_test_loss/self.list_test_loss[0], label="loss")# normalization by the loss of the biggest model
        ax.set_title("Test loss after {} epochs training".format(self.list_epochs[-1]))
        ax.set_xlabel("Global Sparsity")
        ax.set_ylabel("Test Loss")
        ax.legend()
        return ax
