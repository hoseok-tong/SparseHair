import sys
import torch
import numpy as np
from PyQt5 import QtWidgets, QtCore
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from StrandVAE.model.strand_vae import StrandVAE, StrandVAE_1, StrandVAE_2, StrandVAE_3
from StrandVAE.util.state import Stats
from StrandVAE.util.transforms import model_to_tangent_space
from StrandVAE.util.utils import *
from StrandVAE.util.gui_utils import StrandPlotter
from StrandVAE.data.hair20k_dataset import Hair20k
from StrandVAE.model.component.compute_loss import ComputeLossStrandVAE

class ROMControllerGUI(QtWidgets.QWidget):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.scale_factor = 100  # Scale factor to convert slider values to float
        self.load_model()
        self.init_ui()
    
    def load_model(self):
        
        # Display progress dialog while model is loading
        progress = QtWidgets.QProgressDialog('Loading model...', 'Cancel', 0, 100, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.show()
        
        # Select device based on CUDA availability
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load the model architecture from module_arch
        model_params = {
            'dim_in': 1,
            'dim_hidden': 256,
            'dim_out': 6,
            'num_layers': 5,
            'w0_initial': 30.0,
            'latent_dim': 64,
            'coord_length': 99,
        }
        # model = StrandVAE(**model_params)   # baseline
        self.model = StrandVAE_3(**model_params) # ths
        self.model.load_state_dict(torch.load(self.configs['model_path'], map_location=self.device, weights_only=True)["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()        
        progress.setValue(50)
        
        test_data = np.load('/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE/data/usc_hair_resampled/strands00001_mirror.npz')
        test_hairstyle_tangent = torch.from_numpy(test_data['vertsS_tan'])
        test_tbns = torch.from_numpy(test_data['TBNs'])
        test_roots = torch.from_numpy(test_data['roots'])

        self.test_strand = test_hairstyle_tangent[2000].to(self.device)
        save_hair2pc('./test_strand.obj', self.test_strand.reshape(-1,3))
        # print(test_strand.shape)    # (100,3)
        
        self.z, _ = self.model.encode(self.test_strand.unsqueeze(0))
        self.z = torch.cat([self.z, self.model.length_label], dim=1)    # (batch_size, 65)
        self.z = self.model.decoder_input(self.z)
        print(self.z.shape)
        print(self.z[:,-1])
        output = self.model.dec(self.z)  # Generate initial strand output

        self.init_strand = output.detach().cpu().numpy()  # Save initial strand data
        save_hair2pc('./init_strand.obj', self.init_strand.reshape(-1,3))
        progress.setValue(100)
        
    def init_ui(self):
        self.z_update = self.z.clone()

        # Initialize the rendering interface
        renderer = QVTKRenderWindowInteractor()
        self.plotter = StrandPlotter(shape=(1, 1), qt_widget=renderer)
        self.plotter.show(axes=4)
        self.plotter.init_strand('strand_0', 0, *self.init_strand, alpha=1.0)
        self.plotter.add_strand()
        
        # Create vertical layout for sliders
        slider_layout = QtWidgets.QVBoxLayout()
        
        # Create and configure the first slider for latent[:32]
        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setMinimum(-40)
        self.slider1.setMaximum(40)
        self.slider1.setValue(0)
        self.slider1.valueChanged.connect(self.on_latent1_changed)
        slider_layout.addWidget(self.slider1)
        
        # Create and configure the second slider for latent[32:]
        self.slider2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider2.setMinimum(-40)
        self.slider2.setMaximum(40)
        self.slider2.setValue(0)
        self.slider2.valueChanged.connect(self.on_latent2_changed)
        slider_layout.addWidget(self.slider2)
        
        # Wrap slider layout in a widget
        slider_widget = QtWidgets.QWidget()
        slider_widget.setLayout(slider_layout)
        
        # Create main layout and add widgets
        layout = QtWidgets.QGridLayout(self)
        layout.addWidget(renderer, 0, 0)  # Renderer at (0, 0)
        layout.addWidget(slider_widget, 0, 1)  # Sliders at (0, 1)
        layout.setColumnStretch(0, 3)  # More space for renderer
        layout.setColumnStretch(1, 1)  # Less space for sliders
        self.setLayout(layout)

    def on_latent1_changed(self):
        # Update the first 32 latent values when the first slider is changed
        value = self.slider1.value() / self.scale_factor
        # print(self.z.shape) # (1, 64)
        # print(self.z_update.shape) # (1, 64)
        # self.z_update[:, -1] = self.z[:, -1] + torch.tensor(value).to(self.device)

        length_label = self.model.length_label + torch.tensor(value).to(self.device)
        z, _ = self.model.encode(self.test_strand.unsqueeze(0))
        z = torch.cat([z, length_label], dim=1)    # (batch_size, 65)
        print(z[:,-1])
        self.z_update = self.model.decoder_input(z)
        self.update_plot()

    def on_latent2_changed(self):
        # Update the last 32 latent values when the second slider is changed
        value = self.slider2.value() / self.scale_factor
        # self.z_update[:, 32:] = self.z[:, 32:] * torch.tensor(0).to(self.device)
        self.z_update[:, :-1] = self.z[:, :-1] + torch.tensor(value).to(self.device)
        self.update_plot()

    def update_plot(self):
        # Generate new strand output and update the plot
        output = self.model.dec(self.z_update)
        strand = output.detach().cpu().numpy()
        self.plotter.update(strand)
        self.test_strand = output


if __name__ == '__main__':
    configs = {
        'model_path': '/hdd_sda1/tonghs/workspace/SparseHair/StrandVAE/output/strand_vae/2024-10-22_14-09-13/checkpoint_epoch_1400.pth',
    }
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('fusion')
    window = ROMControllerGUI(configs)
    window.setWindowTitle('Refiner GUI')
    window.setGeometry(100, 200, 1400, 1000)
    window.show()
    sys.exit(app.exec_())
