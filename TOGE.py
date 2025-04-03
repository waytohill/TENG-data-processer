import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec
import numpy as np
import scipy.signal
import os

class ResponsivePlot:
    def __init__(self, master):
        self.master = master
        self.figure = plt.figure(figsize=(10, 8), dpi=100)
        self._setup_layout()
        self._create_canvas()
        self._bind_events()

    def _setup_layout(self):
        self.gs = GridSpec(3, 2, figure=self.figure,
                          left=0.1, right=0.98,
                          bottom=0.1, top=0.95,
                          hspace=0.5, wspace=0.3)
        self.axes = [self.figure.add_subplot(pos) for pos in [
            self.gs[0, 0], self.gs[0, 1],
            self.gs[1, 0], self.gs[1, 1],
            self.gs[2, 0], self.gs[2, 1]
        ]]
        for ax in self.axes[2:]:
            ax.sharex(self.axes[0])

    def _create_canvas(self):
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master, pack_toolbar=False)
        self.toolbar.grid(row=0, column=0, sticky='ew')
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky='nsew')

    def _bind_events(self):
        self.canvas.get_tk_widget().bind('<Configure>', self._on_resize)

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
        except:
            pass
        self.canvas.draw_idle()

    def update_plots(self, processor):
        plots_config = [
            (0, "原始信号", [processor.ydata], {'color': '#39C5BB'}),
            (1, "基线对比", 
             [processor.med_baseline, processor.ave_baseline],
             {'labels': ['中值基线', '均值基线']}),
            (2, "中值滤波", [processor.med_filt], {}),
            (3, "均值滤波", [processor.ave_filt], {}),
            (4, "中值处理流程", 
             [processor.med_filt, processor.med_savgol, processor.final_med],
             {'alpha': [1, 0.6, 0.3]}),
            (5, "均值处理流程", 
             [processor.ave_filt, processor.ave_savgol, processor.final_ave],
             {'alpha': [1, 0.6, 0.3]})
        ]

        for idx, title, datasets, styles in plots_config:
            ax = self.axes[idx]
            ax.clear()
            ax.set_title(title, fontsize=10)
            for i, data in enumerate(datasets):
                alpha = styles.get('alpha', [1]*len(datasets))[i]
                label = styles.get('labels', [None]*len(datasets))[i]
                ax.plot(processor.xdata, data, 
                       alpha=alpha, 
                       label=label,
                       linewidth=1 if idx>3 else 1.2)
            if idx in [1,4,5]:
                ax.legend(fontsize=8, loc='upper right')
        
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
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
        self.root.title("通山街数据处理器 v1.2")
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

        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=0, column=0, sticky='nsew')
        self.plot_area = ResponsivePlot(plot_frame)
        
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel")
        control_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        self._create_controls(control_frame)

    def _create_controls(self, parent):
        self.sliders = {}
        param_config = [
            ('freq', "频率 (Hz)", 0.1, 8.0, 0.1),
            ('sample_rate', "采样率", 50, 1000, 10),
            ('window_size', "窗口大小", 5, 100, 1),
            ('savgol_window', "SG 窗口", 5, 99, 2),
            ('savgol_order', "SG 阶数", 1, 5, 1)
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

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill='x', pady=10)
        
        ttk.Button(btn_frame, text="打开文件", command=self._open_file).pack(fill='x', pady=3)
        ttk.Button(btn_frame, text="重新计算", command=self._recalculate).pack(fill='x', pady=3)
        ttk.Button(btn_frame, text="保存图像", command=self._save_image).pack(fill='x', pady=3)
        ttk.Button(btn_frame, text="保存数据", command=self._save_data).pack(fill='x', pady=3)
        ttk.Button(btn_frame, text="退出程序", command=self._safe_exit).pack(fill='x', pady=3)

    def _on_param_change(self, param, value):
        """新增参数变更处理方法"""
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
        filename = filedialog.askopenfilename(filetypes=[("CSV文件", "*.csv")])
        if filename and self.processor.load_data(filename):
            self.processor.process_data()
            self.plot_area.update_plots(self.processor)

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
                filetypes=[("CSV文件", "*.csv")],
                title="选择要覆盖的原始文件"
            )
        else:
            default_name = "processed_" + self.processor.data.attrs.get('filename', 'data.csv')
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV文件", "*.csv")],
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