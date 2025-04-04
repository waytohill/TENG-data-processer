import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec
import numpy as np
import scipy.signal
import os
from scipy.interpolate import UnivariateSpline

versionNumber = "V0.1.3.20250404"


def compute_envelope(signal, x, smoothing_factor=None):
    valid = np.isfinite(signal)
    if not np.any(valid):
        return np.full_like(x, np.nan), np.full_like(x, np.nan), np.full_like(x, np.nan)

    default_s = None

    peaks = scipy.signal.find_peaks(signal)[0]
    if len(peaks) < 2:
        upper = np.interp(x, x[valid], signal[valid])
    else:
        if smoothing_factor is None:
            default_s = 0.1 * len(peaks)
        else:
            default_s = smoothing_factor
        spline_upper = UnivariateSpline(x[peaks], signal[peaks], s=default_s)
        upper = spline_upper(x)

    throughs = scipy.signal.find_peaks(-signal)[0]

    if len(throughs) < 2:
        lower = np.interp(x, x[valid], signal[valid])
    else:
        if smoothing_factor is None:
            default_s = 0.1 * len(throughs)
        else:
            default_s = smoothing_factor
        spline_lower = UnivariateSpline(x[throughs], signal[throughs], s=default_s)
        lower = spline_lower(x)

    diff = upper - lower
    return upper, lower, diff

class ResponsivePlot:
    def __init__(self, master):
        self.master = master
        self.figure = plt.figure(figsize=(10, 8), dpi=100)
        self._setup_layout()
        self._create_canvas()
        self._bind_events()
        self.sync_all = False
        self._updating = False
        self.sync_indices = [True]*len(self.axes)

        for ax in self.axes:
            ax.callbacks.connect('xlim_changed', self._on_xlim_changed)
            

    def _setup_layout(self):
        self.gs = GridSpec(4, 2, figure=self.figure,
                          left=0.1, right=0.98,
                          bottom=0.1, top=0.95,
                          hspace=0.5, wspace=0.3)
        self.axes = [self.figure.add_subplot(pos) for pos in [
            self.gs[0, 0], self.gs[0, 1],
            self.gs[1, 0], self.gs[1, 1],
            self.gs[2, 0], self.gs[2, 1],
            self.gs[3, 0], self.gs[3, 1]
        ]]

        # share axes
        #for ax in self.axes[2:]:
        #    ax.sharex(self.axes[0])

    def _create_canvas(self):
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master, pack_toolbar=False)
        self.toolbar.grid(row=0, column=0, sticky='ew')
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky='nsew')

    def _bind_events(self):
        self.canvas.get_tk_widget().bind('<Configure>', self._on_resize)

    def update_sync_indices(self, new_sync_list):
        self.sync_indices = new_sync_list

    def _on_xlim_changed(self, changed_ax):

        if not self.sync_all or self._updating:
            return
        
        self._updating = True
        new_xlim = changed_ax.get_xlim()

        for idx, ax in enumerate(self.axes):
            if ax is not changed_ax and self.sync_indices[idx]:
                ax.set_xlim(new_xlim)
        self.canvas.draw_idle()
        self._updating = False
            

    def _on_resize(self, event):
        if event.width < 1000 or event.height < 700:
            return
        
        toolbar_height = self.toolbar.winfo_height()
        available_height = event.height - toolbar_height - 20
        
        self.figure.set_size_inches(
            event.width / self.figure.dpi,
            available_height / self.figure.dpi
        )
        
        width_ratio = min(1.0, event.width / 1500)
        height_ratio = min(1.0, event.height / 1000)
        
        self.gs.update(
            left=0.08 + 0.02 * width_ratio,
            right=0.98 - 0.02 * width_ratio,
            hspace=0.4 + 0.2 * height_ratio,
            wspace=0.25 + 0.1 * width_ratio
        )
        
        try:
            self.figure.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()

    def update_plots(self, processor):
        plots_config = [
            (0, "Origin Signal", [processor.ydata], {'color': ['#39C5BB']}),
            (1, "Baseline", 
             [processor.med_baseline, processor.ave_baseline],
             {'labels': ['Mid Baseline', 'Ave Baseline']}),
            (2, "Med Filter", [processor.med_filt], {}),
            (3, "Ave Filter", [processor.ave_filt], {}),
            (4, "Med Processor", 
             [processor.med_filt, processor.med_savgol, processor.final_med],
             {'alpha': [0.3, 0.7, 1], 'color': ['#1772b4','#f17666', '#b7d07a']}),
            (5, "Ave Processor", 
             [processor.ave_filt, processor.ave_savgol, processor.final_ave],
             {'alpha': [0.3, 0.7, 1], 'color': ['#1772b4','#f17666', '#b7d07a']})
        ]

        for idx, title, datasets, styles in plots_config:
            ax = self.axes[idx]
            ax.clear()
            ax.set_title(title, fontsize=10)

            alpha_list = styles.get('alpha', [1]*len(datasets))
            label_list = styles.get('labels', [None]*len(datasets))
            color_list = styles.get('color', [None]*len(datasets))

            for data, al, cl, ll in zip(datasets, alpha_list, color_list, label_list):
                ax.plot(processor.xdata, data, 
                       alpha=al, 
                       label=ll,
                       color=cl,
                       linewidth=1 if idx > 3 else 1.2)
            

            ax.legend(fontsize=8, loc='best')
            #ax.axvline()
            ax.autoscale_view()
        
        # merged into plots_config
        self.update_fft_plots(processor)
        self.canvas.draw()
    
    def update_fft_plots(self, processor):
        sample_rate = processor.params['sample_rate'][0]

        valid_idx = np.where(np.isfinite(processor.final_med))[0]
        if len(valid_idx) == 0:
            return
        
        start_idx = valid_idx[0]
        end_idx = valid_idx[-1] + 1

        valid_final_med = processor.final_med[start_idx:end_idx]
        valid_final_ave = processor.final_ave[start_idx:end_idx]
        N = len(valid_final_med)

        if N == 0:
            return

        fft_med = np.fft.fft(valid_final_med)
        fft_ave = np.fft.fft(valid_final_ave)

        freq = np.fft.fftfreq(N, d=1.0/sample_rate)

        pos_mask = freq >= 0
        freq_pos = freq[pos_mask]
        fft_med_pos = np.abs(fft_med)[pos_mask]
        fft_ave_pos = np.abs(fft_ave)[pos_mask]

        if len(fft_med_pos) > 1:
            idx_med = np.argmax(fft_med_pos[1:]) + 1
        else:
            idx_med = 0

        dom_freq_med = freq_pos[idx_med]

        if len(fft_ave_pos) > 1:
            idx_ave = np.argmax(fft_ave_pos[1:]) + 1
        else:
            idx_ave = 0
        
        dom_freq_ave = freq_pos[idx_ave]

        ax_med = self.axes[6]
        ax_med.clear()
        ax_med.plot(freq_pos, fft_med_pos, color='#a4aca7')
        ax_med.set_title("FFT of Final Med", fontsize=10)
        ax_med.set_xlabel("Frequency (Hz)")
        ax_med.set_ylabel("Amplitude")
        ax_med.axvline(x=dom_freq_med, color='#f9cb8b', linestyle='--', linewidth=1, alpha = 0.8, label=f"Dom Freq: {dom_freq_med:.2f} Hz")
        ax_med.legend(fontsize=8, loc='best')

        ax_ave = self.axes[7]
        ax_ave.clear()
        ax_ave.plot(freq_pos, fft_ave_pos, color='#b5aa90')
        ax_ave.set_title("FFT of Final Ave", fontsize=10)
        ax_ave.set_xlabel("Frequency (Hz)")
        ax_ave.set_ylabel("Amplitude")
        ax_ave.axvline(x=dom_freq_ave, color='#f17666', linestyle='--', linewidth=1, alpha=0.8, label=f"Dom Freq: {dom_freq_ave:.2f} Hz")
        ax_ave.legend(fontsize=8, loc='best')

# modified later
class EnvelopePlot:
    def __init__(self, master):
        self.master = master
        self.figure = plt.figure(figsize=(10,8), dpi=100)
        self._setup_layout()
        self._create_canvas()

    def _setup_layout(self):
        self.axes = [self.figure.add_subplot(4, 1, i+1) for i in range(4)]
    
        
    def _create_canvas(self):
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master, pack_toolbar=False)
        self.toolbar.grid(row=0, column=0, sticky='ew')
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky='nsew')
        
        
    def update_plots(self, processor):
        x = processor.xdata

        up_med, low_med, diff_med = compute_envelope(processor.med_filt, x)
        up_ave, low_ave, diff_ave = compute_envelope(processor.ave_filt, x)

        self.axes[0].clear()
        self.axes[0].plot(x[processor.trim:len(x)-processor.trim], up_med[processor.trim:len(x)-processor.trim], 'b--', label='Upper Envelope')
        self.axes[0].plot(x[processor.trim:len(x)-processor.trim], low_med[processor.trim:len(x)-processor.trim], 'g--', label='Lower Envelope')
        self.axes[0].plot(x[processor.trim:len(x)-processor.trim], diff_med[processor.trim:len(x)-processor.trim], 'r-', label='Difference')
        self.axes[0].set_title("Envelope of Med Filter")
        self.axes[0].legend(fontsize=8)

        self.axes[1].clear()
        self.axes[1].plot(x[processor.trim:len(x)-processor.trim], up_ave[processor.trim:len(x)-processor.trim], 'b--', label='Upper Envelope')
        self.axes[1].plot(x[processor.trim:len(x)-processor.trim], low_ave[processor.trim:len(x)-processor.trim], 'g--', label='Lower Envelope')
        self.axes[1].plot(x[processor.trim:len(x)-processor.trim], diff_ave[processor.trim:len(x)-processor.trim], 'r-', label='Difference')
        self.axes[1].set_title("Envelope of Ave Filter")
        self.axes[1].legend(fontsize=8)

        self.axes[2].clear()
        self.axes[2].plot(x[processor.trim:len(x)-processor.trim], diff_med[processor.trim:len(x)-processor.trim], 'r-', label='Difference')
        self.axes[2].set_title("Envelope of Med Filter")
        self.axes[2].legend(fontsize=8)

        self.axes[3].clear()
        self.axes[3].plot(x[processor.trim:len(x)-processor.trim], diff_ave[processor.trim:len(x)-processor.trim], 'r-', label='Difference')
        self.axes[3].set_title("Envelope of Ave Filter")
        self.axes[3].legend(fontsize=8)

        self.canvas.draw()


class OptimizedProcessor:
    def __init__(self):
        self._init_parameters()
        self.data = None
        self.processed_columns = {
            'med_baseline': None,
            'ave_baseline': None,
            'med_filt': None,
            'ave_filt': None,
            'med_savgol': None,
            'ave_savgol': None,
            'final_med': None,
            'final_ave': None
        }

    def _init_parameters(self):
        self.params = {
            'freq': (0.6, 0.1, 8.0),
            'sample_rate': (500, 50, 1000),
            'window_size': (20, 5, 100),
            'savgol_window': (53, 5, 99),
            'savgol_order': (3, 1, 5)
        }

    def load_data(self, filename):
        try:
            self.data = pd.read_csv(filename, engine='c')
            self.data.attrs['filename'] = os.path.basename(filename)
            self._preprocess_data()
            return True
        except Exception as e:
            messagebox.showerror("错误", f"文件读取失败: {str(e)}")
            return False

    """need to be modified"""
    def _preprocess_data(self):
        if 'time' not in self.data.columns:
            self.data.insert(0, 'time', np.arange(len(self.data)) * 0.002)
        self.xdata = self.data['time'].values
        self.ydata = self.data.iloc[:, 1].values.astype(np.float64)

    def process_data(self):
        kernel_size = int(self.params['freq'][0] * self.params['sample_rate'][0]) + 1
        window_size = int(self.params['window_size'][0])
        savgol_window = int(self.params['savgol_window'][0])
        savgol_order = int(self.params['savgol_order'][0])
        
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        savgol_window = savgol_window + 1 if savgol_window % 2 == 0 else savgol_window

        self.med_baseline = scipy.signal.medfilt(self.ydata, kernel_size)
        window = np.ones(kernel_size)/kernel_size
        self.ave_baseline = np.convolve(self.ydata, window, 'same')
        self.med_filt = self.ydata - self.med_baseline
        self.ave_filt = self.ydata - self.ave_baseline

        half_k = kernel_size // 2

        self.med_filt[:half_k] = np.nan
        self.med_filt[-half_k:] = np.nan
        self.ave_filt[:half_k] = np.nan
        self.ave_filt[-half_k:] = np.nan

        self.trim = half_k

        

        self.med_savgol = scipy.signal.savgol_filter(
            self.med_filt, 
            savgol_window,
            savgol_order,
            mode='nearest'
        )
        self.ave_savgol = scipy.signal.savgol_filter(
            self.ave_filt,
            savgol_window,
            savgol_order,
            mode='nearest'
        )
        smooth_window = np.ones(window_size)/window_size
        self.final_med = np.convolve(self.med_savgol, smooth_window, 'same')
        self.final_ave = np.convolve(self.ave_savgol, smooth_window, 'same')
        
        self.processed_columns.update({
            'med_baseline': self.med_baseline,
            'ave_baseline': self.ave_baseline,
            'med_filt': self.med_filt,
            'ave_filt': self.ave_filt,
            'med_savgol': self.med_savgol,
            'ave_savgol': self.ave_savgol,
            'final_med': self.final_med,
            'final_ave': self.final_ave
        })

    def save_processed_data(self, filename):
        try:
            save_data = self.data.copy()
            for col_name, data in self.processed_columns.items():
                if data is not None and len(data) == len(save_data):
                    save_data[col_name] = data
            save_data.to_csv(filename, index=False)
            return True
        except Exception as e:
            messagebox.showerror("保存错误", f"数据保存失败: {str(e)}")
            return False

class AdaptiveGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.processor = OptimizedProcessor()
        self._setup_main_window()
        self._create_widgets()
        self.timer_id = None

    def _setup_main_window(self):
        self.root.title("TENG Data Processor " + versionNumber)
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        main_frame.columnconfigure(0, weight=4)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=0, column=0, sticky='nsew')

        main_plots_frame = ttk.Frame(self.notebook)
        self.notebook.add(main_plots_frame, text="Main Plots")
        self.plot_area = ResponsivePlot(main_plots_frame)

        envelope_plots_frame = ttk.Frame(self.notebook)
        self.notebook.add(envelope_plots_frame, text="Envelope Plots")
        self.envelope_plot = EnvelopePlot(envelope_plots_frame)



        #plot_frame = ttk.Frame(main_frame)
        #plot_frame.grid(row=0, column=0, sticky='nsew')
        #self.plot_area = ResponsivePlot(plot_frame)
        
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel")
        control_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        self.file_label = ttk.Label(control_frame, text="File: N/A")
        self.file_label.pack(fill='x', pady=4)


        self._create_controls(control_frame)

    def _create_controls(self, parent):
        self.sliders = {}
        param_config = [
            ('freq', "Frequency (Hz)", 0.1, 8.0, 0.1),
            ('sample_rate', "Sample rate", 50, 1000, 10),
            ('window_size', "Window size", 5, 100, 1),
            ('savgol_window', "SG Window", 5, 99, 2),
            ('savgol_order', "SG order", 1, 5, 1)
        ]

        for param, label_text, min_, max_, step in param_config:
            frame = ttk.Frame(parent)
            frame.pack(fill='x', pady=4)
            
            label = ttk.Label(frame, text=f"{label_text}:")
            label.pack(side='left', padx=2)
            
            validate_num = (frame.register(lambda s: s.replace('.','',1).isdigit() or s == ""), '%S')
            entry = ttk.Entry(frame, width=6, validate="key", 
                            validatecommand=validate_num)
            
            slider = ttk.Scale(frame, from_=min_, to=max_,
                              command=lambda v, p=param: self._on_param_change(p, v))
            
            entry.insert(0, str(self.processor.params[param][0]))
            slider.set(self.processor.params[param][0])
            entry.bind('<KeyRelease>', 
                      lambda e, p=param, s=slider: self._entry_update(p, s, e))
            slider.config(command=lambda v, p=param, e=entry: 
                         self._slider_update(p, e, v))
            
            entry.pack(side='left', padx=2)
            slider.pack(side='right', fill='x', expand=True)
            self.sliders[param] = (slider, entry)

        self.sync_var = tk.BooleanVar()
        sync_check = ttk.Checkbutton(parent, text="Sync x-limits across subplots",
                                     variable=self.sync_var,
                                     command=self._toggle_sync)
        sync_check.pack(pady=5)

        subplots_frame = ttk.LabelFrame(parent, text="Apply x-limit to subplots:")
        subplots_frame.pack(fill='x', pady=5)
        self.subplot_sync_vars = {}
        for idx in range(len(self.plot_area.axes)):
            var = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(subplots_frame, text=f"Subplot {idx}", variable=var,
                                   command=self._update_sync_indices)
            chk.pack(anchor='w', padx=4)
            self.subplot_sync_vars[idx] = var


        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill='x', pady=10)
        
        ttk.Button(btn_frame, text="打开文件", command=self._open_file).pack(fill='x', pady=3)
        ttk.Button(btn_frame, text="重新计算", command=self._recalculate).pack(fill='x', pady=3)
        ttk.Button(btn_frame, text="保存图像", command=self._save_image).pack(fill='x', pady=3)
        ttk.Button(btn_frame, text="保存数据", command=self._save_data).pack(fill='x', pady=3)
        ttk.Button(btn_frame, text="退出程序", command=self._safe_exit).pack(fill='x', pady=3)

    def _update_sync_indices(self):
        new_sync = [self.subplot_sync_vars[idx].get() for idx in range(len(self.plot_area.axes))]
        self.plot_area.update_sync_indices(new_sync)
    def _toggle_sync(self):
        self.plot_area.sync_all = self.sync_var.get()
        if self.plot_area.sync_all:

            
            xlim = self.plot_area.axes[0].get_xlim()
            self.plot_area._updating = True
            for idx, ax in enumerate(self.plot_area.axes):
                if self.subplot_sync_vars[idx].get():
                    ax.set_xlim(xlim)
            self.plot_area._updating = False
            self.plot_area.canvas.draw_idle()

    def _on_param_change(self, param, value):
        
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
        self.timer_id = self.root.after(500, self._recalculate)

    def _entry_update(self, param, slider, event):
        try:
            value = float(event.widget.get())
            min_val = self.processor.params[param][1]
            max_val = self.processor.params[param][2]
            if min_val <= value <= max_val:
                slider.set(value)
        except ValueError:
            pass

    def _slider_update(self, param, entry, value):
        entry.delete(0, tk.END)
        if param == 'freq':
            entry.insert(0, f"{float(value):.1f}")
        else:
            entry.insert(0, f"{int(float(value))}")

    def _open_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename and self.processor.load_data(filename):
            self.file_label.config(text=f"File: {self.processor.data.attrs['filename']}")
            self.processor.process_data()
            self.plot_area.update_plots(self.processor)
            self.envelope_plot.update_plots(self.processor)

    def _recalculate(self):
        params = {
            'freq': float(self.sliders['freq'][1].get()),
            'sample_rate': int(float(self.sliders['sample_rate'][1].get())),
            'window_size': int(float(self.sliders['window_size'][1].get())),
            'savgol_window': int(float(self.sliders['savgol_window'][1].get())),
            'savgol_order': int(float(self.sliders['savgol_order'][1].get()))
        }

        for param, value in params.items():
            self.processor.params[param] = (value, 
                                          self.processor.params[param][1],
                                          self.processor.params[param][2])
        
        if self.processor.data is not None:
            self.processor.process_data()
            self.plot_area.update_plots(self.processor)
            self.envelope_plot.update_plots(self.processor)

    def _save_image(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            initialfile="processed_plot")
        if filename:
            self.plot_area.figure.savefig(filename, dpi=300, bbox_inches='tight')

    def _save_data(self):
        if self.processor.data is None:
            messagebox.showwarning("警告", "请先打开并处理数据文件")
            return

        if messagebox.askyesno("保存方式", "是否覆盖原始文件？"):
            filename = filedialog.askopenfilename(
                filetypes=[("CSV files", "*.csv")],
                title="选择要覆盖的原始文件"
            )
        else:
            default_name = "processed_" + self.processor.data.attrs.get('filename', 'data.csv')
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfile=default_name
            )

        if filename:
            success = self.processor.save_processed_data(filename)
            if success:
                messagebox.showinfo("成功", f"数据已成功保存至:\n{filename}")

    def _safe_exit(self):
        plt.close('all')
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = AdaptiveGUI()
    app.run()