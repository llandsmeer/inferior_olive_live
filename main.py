import sys
import json
import scipy.signal
import numpy as np

sys.path.append('/home/llandsmeer/Repos/notyet/iolive')
sys.path.append('/home/llandsmeer/repos/llandsmeer/inferior_olive_live')

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout, QSlider, QLabel, QTextEdit, QCheckBox, QComboBox
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg

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
    I_pulse10ms     =  5.0,
    I_noise_amp     =  10.0
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
        self.setGeometry(100, 100, 800, 800)
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
            slider.setMaximum(500)
            if k in ('I_app', 'I_pulse10ms', 'I_noise_amp'):
                slider.setValue(0)
            else:
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
        # Last row (settings)
        settings_layout = QHBoxLayout()
        #
        reset_button = QPushButton('Reset')
        reset_button.clicked.connect(self.on_reset)
        settings_layout.addWidget(reset_button)
        #
        randomize_button = QPushButton('Randomize')
        randomize_button.clicked.connect(self.on_randomize)
        settings_layout.addWidget(randomize_button)
        #
        self.draw_dropdown = QComboBox()
        self.draw_dropdown.addItems([
            'V(soma)', 'V(dend)', 'V(axon)', 
            'I(k,soma)', 'I(kdr,soma)', 'I(na,soma)', 'I(cal,soma)',
            'I(na,axon)', 'I(k,axon)',
            'I(cah,dend)', 'I(kca,dend)', 'I(h,dend)'
            ])
        self.draw_dropdown.currentTextChanged.connect(self.on_slider_update)
        settings_layout.addWidget(self.draw_dropdown)
        #
        self.export_fmt_dropdown = QComboBox()
        self.export_fmt_dropdown.addItems(['Export Assignments', 'Export JSON', 'Export C++ struct'])
        self.export_fmt_dropdown.currentTextChanged.connect(self.on_slider_update)
        settings_layout.addWidget(self.export_fmt_dropdown)
        #
        (slider_layout_left if i + 1 % 2 == 0 else slider_layout_right).addRow(settings_layout)
        # add a readonly multiline text edit
        self.textedit_params = QTextEdit()
        self.textedit_params.setReadOnly(True)
        self.layout.addWidget(self.textedit_params)
        self.show()

    def on_randomize(self):
        for k in params_default:
            if not k.startswith('g_'):
                continue
            v = np.random.normal(self.sliders[k].value()/self.sliders[k].maximum(), 0.0055)*self.sliders[k].maximum()
            v = int(max(0, min(v, self.sliders[k].maximum())))
            self.sliders[k].setValue(v)
        pass

    def on_reset(self):
        for k, v in params_default.items():
            if k in ('I_app', 'I_pulse10ms', 'I_noise_amp'):
                self.sliders[k].setValue(0)
            else:
                self.sliders[k].setValue(int(self.sliders[k].maximum() * part))
        self.on_slider_update()

    def on_slider_update(self):
        params = {}
        for k, v in params_default.items():
            params[k] = (1/part) * v * self.sliders[k].value() / self.sliders[k].maximum()
            self.slider_labels[k].setText(f'{k} ({params[k]:.3f})')
        self.plot(**params)

    def plot(self, **params):
        np.seterr(all='raise')
        export_params = {k: round(v, 8) for k, v in sorted(params.items()) if k != 'I_pulse10ms'}
        export_fmt = self.export_fmt_dropdown.currentIndex()
        if export_fmt == 0:
            # k=v; k=v; k=v; ...
            s = ';'.join(f'{k}={v}' for k, v in export_params.items())
            self.textedit_params.setText(s)
        elif export_fmt == 1:
            # json export
            self.textedit_params.setText(json.dumps(export_params))
        elif export_fmt == 2:
            # C++ struct export
            s = '\n'.join(f'    double {k} = {v};' for k, v in export_params.items())
            s = f'struct iocell {{\n{s}\n}};'
            self.textedit_params.setText(s)
        else:
            raise ValueError(f'Unknown export format: {export_fmt}')
        try:
            iv_trace = iocell.simulate(
                    skip_initial_transient_seconds=1, sim_seconds=1,
                    record_every=4,
                    **params)
        except Exception as ex:
            self.toplabel.setText(f'{repr(ex)}')
            self.graphWidget.clear()
            return
        finally:
            np.seterr(all='warn')
        (soma_Ik, soma_Ikdr, soma_Ina, soma_Ical, V_soma,
         axon_Ina, axon_Ik, V_axon,
         dend_Icah, dend_Ikca, dend_Ih, V_dend, t) = iv_trace.T
        # get statistics
        idx = scipy.signal.find_peaks(V_soma, distance=5)[0]
        peak_height = V_soma.max()
        idx = idx[abs(V_soma[idx] - peak_height) < 5]
        if len(idx) > 2:
            period = np.diff(t[idx]).mean() / 1000
            freq = 1 / period if period > 0 else 0
        else:
            freq = 0
        amp = V_soma.ptp()
        # plotting
        self.graphWidget.clear()
        pg.setConfigOption('foreground', 'w')
        selected = str(self.draw_dropdown.currentText())
        if selected.startswith('V'):
            if   selected == 'V(soma)': V = V_soma
            elif selected == 'V(axon)': V = V_axon
            elif selected == 'V(dend)': V = V_dend
            self.graphWidget.plot(t, V, color='k')
            self.graphWidget.setRange(xRange=[t[0], t[-1]], yRange=[-100, 100])
            self.graphWidget.setLabels(title='Modified de Gruijl inferior olive model', bottom='Time (ms)', left='Membrane potential (mV)')
        elif selected.startswith('I'):
            if   selected == 'I(k,soma)':   I = soma_Ik
            elif selected == 'I(kdr,soma)': I = soma_Ikdr
            elif selected == 'I(na,soma)':  I = soma_Ina
            elif selected == 'I(cal,soma)': I = soma_Ical
            elif selected == 'I(na,axon)':  I = axon_Ina
            elif selected == 'I(k,axon)':   I = axon_Ik
            elif selected == 'I(cah,dend)': I = dend_Icah
            elif selected == 'I(kca,dend)': I = dend_Ikca
            elif selected == 'I(h,dend)':   I = dend_Ih
            self.graphWidget.plot(t, I, color='k')
            self.graphWidget.setLabels(title='Modified de Gruijl inferior olive model', bottom='Time (ms)', left='Current')
        if np.isclose(params['I_pulse10ms'], 0):
            self.toplabel.setText(f'{freq:.1f} Hz, {amp:.1f} mVpp')
        else:
            self.toplabel.setText(f'{amp:.1f} mVpp')

def main():
    app = QApplication([])
    window = Window()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
