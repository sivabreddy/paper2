# import PySimpleGUI as sg
# import numpy as np
# import matplotlib.pyplot as plt
# from Main import Run
# sg.change_look_and_feel('DarkTeal9')    # look and feel theme
#
# # Designing layout
# layout = [[sg.Text("\t\t\tSelect_dataset"), sg.Combo(['Prostate MRI'],size=(13, 20)),sg.Text("\n")],
#           [sg.Text("\t\t\tSelect            "),sg.Combo(["TrainingData(%)","k-fold"], size=(13, 20)), sg.Text(""), sg.InputText(size=(10, 20), key='1'),sg.Button("START", size=(10, 2))],[sg.Text('\n')],
#           [sg.Text("\t\t\t   DCNN\t\t    Panoptic model\t\t    Focal-Net\t\t ResNet\t\t           HFGSO DRN\t\tLHFGSO DMO")],
#           [sg.Text('\tAccuracy '), sg.In(key='11',size=(20,20)), sg.In(key='12',size=(20,20)), sg.In(key='13',size=(20,20)), sg.In(key='14',size=(20,20)),sg.In(key='15',size=(20,20)),sg.In(key='16',size=(20,20)),sg.Text("\n")],
#           [sg.Text('\tSensitivity'), sg.In(key='21',size=(20,20)), sg.In(key='22',size=(20,20)), sg.In(key='23',size=(20,20)), sg.In(key='24',size=(20,20)),sg.In(key='25',size=(20,20)),sg.In(key='26',size=(20,20)), sg.Text("\n")],
#           [sg.Text('\tSpecificity'), sg.In(key='31', size=(20, 20)), sg.In(key='32', size=(20, 20)),sg.In(key='33', size=(20, 20)), sg.In(key='34', size=(20, 20)),sg.In(key='35', size=(20, 20)),sg.In(key='36', size=(20, 20)), sg.Text("\n")],
#           [sg.Text('\t\t\t\t\t\t\t\t\t\t\t\t            '), sg.Button('Run Graph'), sg.Button('CLOSE')]]
#
#
# # to plot graphs
# def plot_graph(result_1, result_2, result_3):
#     plt.figure(dpi=120)
#     loc, result = [], []
#     result.append(result_1)  # appending the result
#     result.append(result_2)
#     result.append(result_3)
#     result = np.transpose(result)
#
#     # labels for bars
#     labels = ['DCNN', 'Panoptic model', 'Focal-Net','ResNet','Proposed HFGSO-based DRN ','proposed']  # x-axis labels ############################
#     tick_labels = ['Accuracy', 'Sensitivity','Specificity']  #### metrics
#     bar_width, s = 0.15, 0.025  # bar width, space between bars
#
#     for i in range(len(result)):  # allocating location for bars
#         if i == 0:  # initial location - 1st result
#             tem = []
#             for j in range(len(tick_labels)):
#                 tem.append(j + 1)
#             loc.append(tem)
#         else:  # location from 2nd result
#             tem = []
#             for j in range(len(loc[i - 1])):
#                 tem.append(loc[i - 1][j] + s + bar_width)
#             loc.append(tem)
#
#     # plotting a bar chart
#     for i in range(len(result)):
#         plt.bar(loc[i], result[i], label=labels[i], tick_label=tick_labels, width=bar_width)
#
#     plt.legend(loc=(0.25, 0.25))# show a legend on the plot -- here legends are metrics
#     plt.show()  # to show the plot
#
#
# # Create the Window layout
# window = sg.Window('142705', layout)
#
# # event loop
# while True:
#     event, values = window.read()  # displays the window
#     if event == "START":
#         if values[1] == 'TrainingData(%)':
#             tp = int(values['1']) / 100
#         else:
#             tp = (int(values['1']) - 1) / int(values['1'])  # k-fold calculation
#         dataset, tr_per = values[0], tp
#         dts = dataset
#         Acc,Sen,Spe = Run.callmain(dts,tr_per)
#
#         window['11'].Update(Acc[0])
#         window['12'].Update(Acc[1])
#         window['13'].Update(Acc[2])
#         window['14'].Update(Acc[3])
#         window['15'].Update(Acc[4])
#         window['16'].Update(Acc[5])
#
#         window['21'].Update(Sen[0])
#         window['22'].Update(Sen[1])
#         window['23'].Update(Sen[2])
#         window['24'].Update(Sen[3])
#         window['25'].Update(Sen[4])
#         window['26'].Update(Sen[5])
#
#         window['31'].Update(Spe[0])
#         window['32'].Update(Spe[1])
#         window['33'].Update(Spe[2])
#         window['34'].Update(Spe[3])
#         window['35'].Update(Spe[4])
#         window['36'].Update(Spe[5])
#
#     if event == 'Run Graph':
#         plot_graph(Acc,Sen,Spe)
#     if event == 'CLOSE':
#         break
#         window.close()
#
#
"""
Tkinter-based GUI for prostate MRI analysis system.
Provides interface to run models and visualize comparative results.
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from Main import Run

class ApplicationGUI:
    """
    Main application window implementing the GUI interface.
    
    Features:
    - Dataset selection
    - Training parameter configuration
    - Model comparison results display
    - Performance visualization
    """
    def __init__(self, root):
        self.root = root
        self.root.title('142705')
        self.root.configure(bg='#004d4d')  # DarkTeal-like color

        # Variables
        self.dataset_var = tk.StringVar()
        self.selection_var = tk.StringVar()
        self.input_var = tk.StringVar()
        self.Acc = None
        self.Sen = None
        self.Spe = None

        # Create frames
        main_frame = tk.Frame(root, bg='#004d4d', padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Row 1: Dataset selection
        row1 = tk.Frame(main_frame, bg='#004d4d')
        row1.pack(fill=tk.X, pady=5)
        tk.Label(row1, text="\t\t\tSelect_dataset", bg='#004d4d', fg='white').pack(side=tk.LEFT)
        dataset_combo = ttk.Combobox(row1, textvariable=self.dataset_var, values=['Prostate MRI'], width=13)
        dataset_combo.pack(side=tk.LEFT, padx=5)
        dataset_combo.current(0)

        # Row 2: Selection type and input
        row2 = tk.Frame(main_frame, bg='#004d4d')
        row2.pack(fill=tk.X, pady=5)
        tk.Label(row2, text="\t\t\tSelect            ", bg='#004d4d', fg='white').pack(side=tk.LEFT)
        selection_combo = ttk.Combobox(row2, textvariable=self.selection_var, values=["TrainingData(%)", "k-fold"], width=13)
        selection_combo.pack(side=tk.LEFT, padx=5)
        selection_combo.current(0)

        input_entry = tk.Entry(row2, textvariable=self.input_var, width=10)
        input_entry.pack(side=tk.LEFT, padx=5)

        start_button = tk.Button(row2, text="START", command=self.start_process, width=10, height=2)
        start_button.pack(side=tk.LEFT, padx=5)

        # Row 3: Headers
        row3 = tk.Frame(main_frame, bg='#004d4d')
        row3.pack(fill=tk.X, pady=10)
        tk.Label(row3, text="\t\t\t   DCNN\t\t    Panoptic model\t\t    Focal-Net\t\t ResNet\t\t           HFGSO DRN\t\tLHFGSO DMO",
                bg='#004d4d', fg='white').pack()

        # Create entries for results
        self.entries = {}
        metrics = ['Accuracy', 'Sensitivity', 'Specificity']

        for i, metric in enumerate(metrics, 1):
            row = tk.Frame(main_frame, bg='#004d4d')
            row.pack(fill=tk.X, pady=5)
            tk.Label(row, text=f'\t{metric}', bg='#004d4d', fg='white').pack(side=tk.LEFT)

            for j in range(1, 7):
                key = f'{i}{j}'
                entry = tk.Entry(row, width=20)
                entry.pack(side=tk.LEFT, padx=2)
                self.entries[key] = entry

        # Row for buttons
        button_row = tk.Frame(main_frame, bg='#004d4d')
        button_row.pack(fill=tk.X, pady=10)
        tk.Label(button_row, text='\t\t\t\t\t\t\t\t\t\t\t\t            ', bg='#004d4d').pack(side=tk.LEFT)

        graph_button = tk.Button(button_row, text='Run Graph', command=self.run_graph)
        graph_button.pack(side=tk.LEFT, padx=5)

        close_button = tk.Button(button_row, text='CLOSE', command=self.root.destroy)
        close_button.pack(side=tk.LEFT, padx=5)

    def start_process(self):
        """
        Handles START button click event.
        Configures training parameters and executes all models.
        Updates UI with accuracy, sensitivity, specificity metrics.
        """
        if self.selection_var.get() == 'TrainingData(%)':
            tp = int(self.input_var.get()) / 100
        else:
            tp = (int(self.input_var.get()) - 1) / int(self.input_var.get())  # k-fold calculation

        dataset, tr_per = self.dataset_var.get(), tp
        dts = dataset
        self.Acc, self.Sen, self.Spe = Run.callmain(dts, tr_per)

        # Update entries with results
        self.entries['11'].delete(0, tk.END)
        self.entries['11'].insert(0, self.Acc[0])
        self.entries['12'].delete(0, tk.END)
        self.entries['12'].insert(0, self.Acc[1])
        self.entries['13'].delete(0, tk.END)
        self.entries['13'].insert(0, self.Acc[2])
        self.entries['14'].delete(0, tk.END)
        self.entries['14'].insert(0, self.Acc[3])
        self.entries['15'].delete(0, tk.END)
        self.entries['15'].insert(0, self.Acc[4])
        self.entries['16'].delete(0, tk.END)
        self.entries['16'].insert(0, self.Acc[5])

        self.entries['21'].delete(0, tk.END)
        self.entries['21'].insert(0, self.Sen[0])
        self.entries['22'].delete(0, tk.END)
        self.entries['22'].insert(0, self.Sen[1])
        self.entries['23'].delete(0, tk.END)
        self.entries['23'].insert(0, self.Sen[2])
        self.entries['24'].delete(0, tk.END)
        self.entries['24'].insert(0, self.Sen[3])
        self.entries['25'].delete(0, tk.END)
        self.entries['25'].insert(0, self.Sen[4])
        self.entries['26'].delete(0, tk.END)
        self.entries['26'].insert(0, self.Sen[5])

        self.entries['31'].delete(0, tk.END)
        self.entries['31'].insert(0, self.Spe[0])
        self.entries['32'].delete(0, tk.END)
        self.entries['32'].insert(0, self.Spe[1])
        self.entries['33'].delete(0, tk.END)
        self.entries['33'].insert(0, self.Spe[2])
        self.entries['34'].delete(0, tk.END)
        self.entries['34'].insert(0, self.Spe[3])
        self.entries['35'].delete(0, tk.END)
        self.entries['35'].insert(0, self.Spe[4])
        self.entries['36'].delete(0, tk.END)
        self.entries['36'].insert(0, self.Spe[5])

    def run_graph(self):
        if self.Acc and self.Sen and self.Spe:
            self.plot_graph(self.Acc, self.Sen, self.Spe)

    def plot_graph(self, result_1, result_2, result_3):
        """
        Generates comparative bar chart of model performance metrics.
        
        Args:
            result_1: Accuracy values for all models
            result_2: Sensitivity values for all models
            result_3: Specificity values for all models
        """
        plt.figure(dpi=120)
        loc, result = [], []
        result.append(result_1)  # appending the result
        result.append(result_2)
        result.append(result_3)
        result = np.transpose(result)

        # labels for bars
        labels = ['DCNN', 'Panoptic model', 'Focal-Net', 'ResNet', 'Proposed HFGSO-based DRN', 'proposed']
        tick_labels = ['Accuracy', 'Sensitivity', 'Specificity']
        bar_width, s = 0.15, 0.025  # bar width, space between bars

        for i in range(len(result)):  # allocating location for bars
            if i == 0:  # initial location - 1st result
                tem = []
                for j in range(len(tick_labels)):
                    tem.append(j + 1)
                loc.append(tem)
            else:  # location from 2nd result
                tem = []
                for j in range(len(loc[i - 1])):
                    tem.append(loc[i - 1][j] + s + bar_width)
                loc.append(tem)

        # plotting a bar chart
        for i in range(len(result)):
            plt.bar(loc[i], result[i], label=labels[i], tick_label=tick_labels, width=bar_width)

        plt.legend(loc=(0.25, 0.25))  # show a legend on the plot
        plt.show()  # to show the plot

if __name__ == "__main__":
    root = tk.Tk()
    app = ApplicationGUI(root)
    root.mainloop()
