import sys
import numpy as np
from PyQt5.Qt import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QSpinBox, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create the sliders
        self.slider1 = QSpinBox()
        self.slider2 = QSpinBox()

        # Set the minimum and maximum values for the sliders
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(100)
        self.slider2.setMinimum(0)
        self.slider2.setMaximum(100)

        # Connect the valueChanged signal of the sliders to the update_plot slot
        self.slider1.valueChanged.connect(self.update_plot)
        self.slider2.valueChanged.connect(self.update_plot)

        # Create the matplotlib figure and canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Create a vertical layout to hold the sliders and the canvas
        self.layout = QVBoxLayout()

        # Add the sliders and the canvas to the layout
        self.layout.addWidget(self.slider1)
        self.layout.addWidget(self.slider2)
        self.layout.addWidget(self.canvas)

        # Create a central widget to hold the layout
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)

        # Set the central widget of the main window
        self.setCentralWidget(self.central_widget)

        # Initialize the plot
        self.update_plot()

    def update_plot(self):
        # Get the values of the sliders
        slider1_value = self.slider1.value()
        slider2_value = self.slider2.value()

        # Generate some random data
        data = np.random.rand(slider1_value, slider2_value)

        # Clear the figure
        self.figure.clear()

        # Add a subplot to the figure
        ax = self.figure.add_subplot(1, 1, 1)

        # Plot the data
        ax.imshow(data, cmap='gray')

        # Draw the canvas
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
