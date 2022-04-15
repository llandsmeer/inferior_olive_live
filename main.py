from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout, QSlider, QLabel
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import scipy.signal

import sys
import numpy as np
sys.path.append('/home/llandsmeer/Repos/notyet/iolive')

import iocell

part = 0.2


params_default = dict(
    g_int           =   0.13,    # Cell internal conductance  -- now a parameter (0.13)
    p1              =   0.25,    # Cell surface ratio soma/dendrite
    p2              =   0.15,    # Cell surface ratio axon(hillock)/soma
    g_CaL           =   1.1,     # Calcium T - (CaV 3.1) (0.7)
    g_h             =   0.12,    # H current (HCN) (0.4996)
    g_K_Ca          =  35.0,     # Potassium  (KCa v1.1 - BK) (35)
    g_ld            =   0.01532, # Leak dendrite (0.016)
    g_la            =   0.016,   # Leak axon (0.016)
    g_ls            =   0.016,   # Leak soma (0.016)
    S               =   1.0,     # 1/C_m, cm^2/uF
    g_Na_s          = 150.0,     # Sodium  - (Na v1.6 )
    g_Kdr_s         =   9.0,     # Potassium - (K v4.3)
    g_K_s           =   5.0,     # Potassium - (K v3.4)
    g_CaH           =   4.5,     # High-threshold calcium -- Ca V2.1
    g_Na_a          = 240.0,     # Sodium
    g_K_a           = 240.0,     # Potassium (20)
    V_Na            =  55.0,     # Sodium
    V_K             = -75.0,     # Potassium
    V_Ca            = 120.0,     # Low-threshold calcium channel
    V_h             = -43.0,     # H current
    V_l             =  10.0,     # Leak
    I_app           =   1.0,
    I_spike         =  5.0,
)

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet('''
            background-color: #000000;
            color: #ffffff;
        ''')
        self.init_ui()
        self.on_slider_update()

    def init_ui(self):
        self.sliders = {}
        self.slider_labels = {}
        self.setWindowTitle('IOLive (Figure 1)')
        self.setGeometry(300, 300, 640, 480)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.graphWidget = pg.PlotWidget()
        self.toplabel = QLabel('Loading...')
        self.layout.addWidget(self.toplabel)
        self.layout.addWidget(self.graphWidget)
        slider_layout = QHBoxLayout()
        slider_layout_left = QFormLayout()
        slider_layout_right = QFormLayout()
        self.layout.addLayout(slider_layout)
        slider_layout.addLayout(slider_layout_left)
        slider_layout.addLayout(slider_layout_right)
        for i, (k, v) in enumerate(params_default.items()):
            slider = QSlider()
            slider.setMinimum(0)
            slider.setMaximum(1000)
            slider.setValue(int(slider.maximum() * part))
            slider.setOrientation(Qt.Horizontal)
            slider_label = QLabel(f'{k} ({v})')
            if i % 2 == 0:
                slider_layout_left.addRow(slider_label, slider)
            else:
                slider_layout_right.addRow(slider_label, slider)
            self.sliders[k] = slider
            self.slider_labels[k] = slider_label
            slider.setStyleSheet('''
            QSlider::sub-page:horizontal {
                background-color: #aaaaaa;
            }
            QSlider::add-page:horizontal {
                background-color: #aaaaaa;
            }
            QSlider::handle:horizontal {
                background-color: #ffffff;
            }
            ''')
            slider.valueChanged.connect(self.on_slider_update)
        self.show()

    def on_slider_update(self):
        params = {}
        for k, v in params_default.items():
            params[k] = (1/part) * v * self.sliders[k].value() / self.sliders[k].maximum()
            self.slider_labels[k].setText(f'{k} ({params[k]:.3f})')
        self.plot(**params)

    def plot(self, **params):
        np.seterr(all='raise')
        try:
            iv_trace = iocell.simulate(skip_initial_transient_seconds=1, sim_seconds=1, **params)
        except Exception as ex:
            self.toplabel.setText(f'{repr(ex)}')
            self.graphWidget.clear()
            return
        finally:
            np.seterr(all='warn')
        (soma_Ik, soma_Ikdr, soma_Ina, soma_Ical, V_soma,
         axon_Ina, axon_Ik, V_axon,
         dend_Icah, dend_Ikca, dend_Ih, V_dend, t) = iv_trace.T
        idx = scipy.signal.find_peaks(V_soma)[0]
        if len(idx) > 2:
            period = np.diff(t[idx]).mean() / 1000
            freq = 1 / period if period > 0 else 0
        else:
            freq = 0
        amp = V_soma.ptp()
        self.graphWidget.clear()
        self.graphWidget.plot(t, V_dend, color='k')
        self.graphWidget.setRange(xRange=[t[0], t[-1]], yRange=[-100, 20])
        #self.graphWidget.xlabel('Time (ms)')
        #self.graphWidget.ylabel('Soma potential (mV)')
        self.toplabel.setText(f'{freq:.1f} Hz | {amp:.1f} mV')

def main():
    app = QApplication([])
    window = Window()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
