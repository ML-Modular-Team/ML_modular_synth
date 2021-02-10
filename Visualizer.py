import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, list_pruning_amounts, list_epochs, lists_loss):
        super().__init__()
        self.list_pruning_amounts = list_pruning_amounts  # list of all pruning amounts applied
        self.list_epochs = list_epochs  # list of all epochs computed
        self.lists_loss = lists_loss  # list of list of losses

        for loss_list in self.lists_loss:
            if len(loss_list) > len(self.list_epochs):
                print("Loss lists length does not match the number of epochs (list_epochs)")

        if len(self.list_pruning_amounts) != len(self.lists_loss):
            print("Losses missing in lists_loss (does not match the number of pruning amount applied)")

    def show_training_loss_pruning(self, indexes=None):
        """Plots the loss's evolution for each pruned version of the model
            Loss function corresponding to indexes (a,b) (b excluded)"""
        if indexes is None:
            a, b = 0, len(self.list_pruning_amounts)
        else:
            a, b = indexes
        for i in range(a, b):
            list_loss = self.lists_loss[i]
            plt.plot(self.list_epochs[:len(list_loss)], list_loss, label="Pa = {:.2f}".format(self.list_pruning_amounts[i]))
        plt.title("Training loss on MNIST")
        plt.legend()
        plt.plot()

    def show_score_pruning(self):
        """Plots the evolution of the last training when the pruning ratio increase"""
        list_final_loss = [elt[-1] for elt in self.lists_loss]
        plt.plot(self.list_pruning_amounts, list_final_loss, label="loss")
        plt.title("Training loss after {} epochs".format(self.list_epochs[-1]))
        plt.legend()
        plt.plot()
