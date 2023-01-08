import re
import sys
from functools import cache

import neptune.new as neptune
import numpy as np
import torch
from matplotlib.backends.backend_qt5agg import (FigureCanvas,
                                                NavigationToolbar2QT)
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QRadioButton, QSpinBox,
                             QVBoxLayout, QWidget)


class Loader:
    def __init__(self, project, api_token, run, ):
        self.data = neptune.init_run(
            project,
            api_token,
            run, mode="read-only"
        ).get_structure()

        self.series = sorted([name for name, series in self.data.items()
                              if not isinstance(series, dict)])

    @cache
    def get(self, name):
        path = name.split("/")
        node = self.data
        for elem in path:
            node = node[elem]
        return node.fetch_values()["value"].to_numpy()

    @cache
    def regex(self, regex):
        for name in self.series:
            if not re.fullmatch(regex, name):
                continue
            return self.get(name)

    @cache
    def distribution(self, regex, epoch):
        result = []
        for name in self.series:
            if not re.fullmatch(regex, name):
                continue
            value = self.get(name)
            if epoch not in range(len(value)):
                continue
            result.append(value[epoch])
        return np.array(result)

    @cache
    def layer_distributions(self, regex, epoch):
        result = []
        layer = 0
        while True:
            d = self.distribution(f"{layer}_.*:{regex}.*", epoch)
            if len(d):
                result.append(d)
                layer += 1
            else:
                return result


loader = Loader(project="WideLearning/Titanic",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLC" +
                "JhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NT" +
                "IzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==",
                run="TIT-28")


def squash(x, y):
    r = np.hypot(x, y)
    k = np.log(r * 1e8 + 1) / r
    return k * x, k * y


def scale(x, y):
    return (x - x.mean()) / x.std(), (y - y.mean()) / y.std()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Create widgets
        self.epoch_spinbox = QSpinBox()
        self.epoch_spinbox.setPrefix("Epoch: ")
        self.epoch_spinbox.setFixedWidth(200)
        self.epoch_spinbox.setMaximum(10**9)

        self.layer_spinbox = QSpinBox()
        self.layer_spinbox.setPrefix("Layer: ")
        self.layer_spinbox.setFixedWidth(200)
        self.layer_spinbox.setMaximum(10**9)

        self.weight_button = QRadioButton("w")
        self.gradient_button = QRadioButton("∇")
        self.gradient_button.setChecked(True)
        self.dweight_button = QRadioButton("Δw")

        self.epoch_spinbox.valueChanged.connect(self.update_plot)
        self.layer_spinbox.valueChanged.connect(self.update_plot)
        self.weight_button.toggled.connect(self.update_plot)
        self.gradient_button.toggled.connect(self.update_plot)
        self.dweight_button.toggled.connect(self.update_plot)

        self.canvas = FigureCanvas(Figure())
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # Define layout
        self.main_layout = QVBoxLayout()
        self.input_layout = QHBoxLayout()

        self.main_layout.addLayout(self.input_layout)
        self.main_layout.addWidget(self.canvas)
        self.input_layout.addWidget(self.toolbar)
        self.input_layout.addWidget(self.weight_button)
        self.input_layout.addWidget(self.gradient_button)
        self.input_layout.addWidget(self.dweight_button)
        self.input_layout.addWidget(self.epoch_spinbox)
        self.input_layout.addWidget(self.layer_spinbox)

        self.setLayout(self.main_layout)

        # Initialize the plot
        self.update_plot()

    def resizeEvent(self, event):
        self.update_plot()
        return super().resizeEvent()

    def type_to_plot(self):
        if self.weight_button.isChecked():
            return "weight"
        if self.gradient_button.isChecked():
            return "gradient"
        return "dweight"

    def update_plot(self):
        # Fetch data and clip inputs
        val_loss = loader.get("val/loss")
        train_loss = loader.get("train/loss")
        n_epochs = len(train_loss)
        self.epoch_spinbox.setMaximum(n_epochs - 1)
        epoch = self.epoch_spinbox.value()

        dists = loader.layer_distributions(f"{self.type_to_plot()}:log", epoch)
        n_params = len(dists)
        self.layer_spinbox.setMaximum(n_params - 1)
        layer = self.layer_spinbox.value()

        # Build the plot from scratch
        self.canvas.figure.clear()
        gs = GridSpec(3, 6)

        ax_loss = self.canvas.figure.add_subplot(gs[0, 0:2])
        ax_loss.set_title("Loss")
        ax_loss.plot(val_loss, color="blue", lw=1)
        ax_loss.plot(train_loss, "--", color="blue", lw=1)
        ax_loss.axvline(x=epoch, color="red", lw=0.5)

        ax_avglayer = self.canvas.figure.add_subplot(gs[0, 2:4])
        ax_avglayer.set_title("Layer distributions")
        ax_avglayer.violinplot(dists, positions=range(n_params), showextrema=False, quantiles=[
                               [0.1, 0.5, 0.9] for _ in range(n_params)])
        ax_avglayer.axvline(x=layer, color="red", lw=0.5)
        ax_avglayer.set_ylim(-10, 2)

        ax_coslayer = self.canvas.figure.add_subplot(gs[0, 4:6])
        ax_coslayer.set_title("Layer cosine similarities")
        ax_coslayer.set_ylim(-1, 1)
        ax_coslayer.axvline(x=epoch, color="red", lw=0.5)
        for i in (list(range(0, layer)) + list(range(layer + 1, n_params)) + [layer]):
            c = loader.regex(f"{i}_.*:{self.type_to_plot()}:cosine_sim")
            if c is None:
                break
            if i == layer:
                ax_coslayer.plot(c, "-", lw=0.5, color="red")
            else:
                ax_coslayer.plot(c, "-", lw=0.1, color="blue")

        def plot_path(ax, f, r):
            ax.set_xlim(-r, r)
            ax.set_ylim(-r, r)
            for i in range(n_params):
                x = loader.regex(f"{i}_.*:x")
                y = loader.regex(f"{i}_.*:y")
                x, y = f(x, y)
                if i == layer:
                    ax.plot(x, y, "o-", lw=0.1, ms=1, color="blue")
                    ax.plot([x[epoch]], [y[epoch]], "o", ms=4, color="red")
                else:
                    ax.plot(x, y, "o-", lw=0.1, ms=1, color="blue")
                    ax.plot([x[epoch]], [y[epoch]], "o", ms=4, color="blue")

        ax_scaledpath = self.canvas.figure.add_subplot(gs[1:, 0:3])
        ax_scaledpath.set_title("Scaled layer paths")
        plot_path(ax_scaledpath, f=scale, r=5)

        ax_squashedpath = self.canvas.figure.add_subplot(gs[1:, 3:6])
        ax_squashedpath.set_title("Squashed layer paths")
        plot_path(ax_squashedpath, f=squash, r=25)

        self.canvas.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()

    widget.show()
    sys.exit(app.exec_())
