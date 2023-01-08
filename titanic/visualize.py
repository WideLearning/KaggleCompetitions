import sys

import neptune.new as neptune
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

        self.series = [name for name, series in self.data.items()
                       if not isinstance(series, dict)]
        print(self.series)
        self.fetched = {}

    def get(self, name):
        if name not in self.fetched:
            path = name.split("/")
            node = self.data
            for elem in path:
                node = node[elem]
            self.fetched[name] = node.fetch_values()["value"].to_numpy()
        return self.fetched[name]


loader = Loader(project="WideLearning/Titanic",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLC" +
                "JhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NT" +
                "IzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==",
                run="TIT-16")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Create the input widgets
        self.epoch_spinbox = QSpinBox()
        self.epoch_spinbox.setPrefix("Epoch: ")
        self.epoch_spinbox.setFixedWidth(200)
        self.epoch_spinbox.setMaximum(10**9)

        self.layer_spinbox = QSpinBox()
        self.layer_spinbox.setPrefix("Layer: ")
        self.layer_spinbox.setFixedWidth(200)
        self.layer_spinbox.setMaximum(10**9)

        self.weight_button = QRadioButton("w")
        self.weight_button.setChecked(True)
        self.gradient_button = QRadioButton("∇")

        # Connect the valueChanged signal of the inputs to the update_plot slot
        self.epoch_spinbox.valueChanged.connect(self.update_plot)
        self.layer_spinbox.valueChanged.connect(self.update_plot)

        # Create the matplotlib figure and canvas
        self.canvas = FigureCanvas(Figure())
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # Create layouts
        self.main_layout = QVBoxLayout()
        self.input_layout = QHBoxLayout()

        # Add the sliders and the canvas to the layout
        self.main_layout.addLayout(self.input_layout)
        self.main_layout.addWidget(self.canvas)
        self.input_layout.addWidget(self.toolbar)
        self.input_layout.addWidget(self.weight_button)
        self.input_layout.addWidget(self.gradient_button)
        self.input_layout.addWidget(self.epoch_spinbox)
        self.input_layout.addWidget(self.layer_spinbox)

        # # Set the central widget of the main window
        # self.setCentralWidget(self.central_widget)
        self.setLayout(self.main_layout)
        # Initialize the plot
        self.update_plot()

    def update_plot(self):
        # Get the values of the sliders
        epoch = self.epoch_spinbox.value()
        layer = self.layer_spinbox.value()

        gs = GridSpec(3, 4)
        self.canvas.figure.clear()

        ax_loss = self.canvas.figure.add_subplot(gs[0, :2])
        ax_loss.set_title("Loss")
        val_loss = loader.get("val/loss")
        ax_loss.plot(val_loss, color="blue", lw=1)
        train_loss = loader.get("train/loss")
        ax_loss.plot(train_loss, "--", color="blue", lw=1)
        ax_loss.axvline(x=epoch, color="red", lw=0.5)

        ax_netpath = self.canvas.figure.add_subplot(gs[0, 2])
        ax_netpath.set_title("Network path")

        ax_layerpath = self.canvas.figure.add_subplot(gs[0, 3])
        ax_layerpath.set_title("Layer path")

        ax_avglayer = self.canvas.figure.add_subplot(gs[1, :2])
        ax_avglayer.set_title("Layer averages")
        ax_avglayer.axvline(x=layer, color="red", lw=0.5)

        ax_coslayer = self.canvas.figure.add_subplot(gs[1, 2:])
        ax_coslayer.set_title("Layer cosine similarities")

        ax_distcur = self.canvas.figure.add_subplot(gs[2, :2])
        ax_distcur.set_title("Distribution for current layer")

        ax_coscur = self.canvas.figure.add_subplot(gs[2, 2:])
        ax_coscur.set_title("Cosine similarities for current layer")

        self.canvas.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec_())
