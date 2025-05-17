import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# to plot graph
def plot_graph(result_1, result_2, result_3, result_4,x_lab,y_lab,tick_labels1,filename):
    colorr = ['#7D3F98','#EEBBF4','#AD6CC1','#632E82','#351458','#FEEED2']
    plt.figure()        # dpi (dots-per-inch) = 150
    loc, result = [], []
    result.append(result_1)  # appending the result
    result.append(result_2)
    result.append(result_3)
    result.append(result_4)

    result = np.transpose(result)

    # labels for bars
    labels = ['DCNN', 'Panoptic model ', 'Focal-Net','ResNet ', 'HFGSO based DRN ','Proposed  LHFGSO-DMN']  # x-axis labels
    tick_labels = tick_labels1  # metrics
    bar_width, s = 0.14, 0.0  # bar width, space between bars

    for i in range(len(result)):  # allocating location for bars
        if i is 0:  # initial location - 1st result
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
        plt.bar(loc[i], result[i],color = colorr[i], label=labels[i], tick_label=tick_labels, width=bar_width, edgecolor='black')
    plt.subplots_adjust(bottom=0.3)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2,fontsize=11)
    # plt.legend()  # show a legend on the plot
    plt.xlabel(x_lab) ################## Delay
    plt.ylabel(y_lab)          ################## MSE, RMSE
    # plt.legend()
    plt.savefig(filename + ".jpg")
    plt.show()
    # plt.legend(loc=(0.23, 0.06))     # loc=(0.25, 0.25) ---- 0.55, 0.7
    # plt.savefig(filename+".jpg")
    #plt.show()  # to show the plot

# data = pd.read_excel('Acc_tr_ca.xlsx',header=None)
# data = np.array(data)
data = np.load('ACC_tr_ca.npy')
result_1, result_2, result_3, result_4 = data[0], data[1], data[2], data[3]
y_lab = 'Accuracy'
x_lab = 'Training data(%)'
tick_labels1 = ['60', '70', '80', '90']
filename = 'ACC_tr_ca'
plot_graph(result_1, result_2, result_3, result_4,x_lab,y_lab,tick_labels1,filename)

data = np.load('SEN_tr_ca.npy')
# data = pd.read_excel('sen_tr_ca.xlsx',header=None)
# data = np.array(data)
result_1, result_2, result_3, result_4 = data[0], data[1], data[2], data[3]
y_lab = 'Sensitivity'
x_lab = 'Training data(%)'
tick_labels1 = ['60', '70', '80', '90']
filename = 'Sen_tr_ca'
plot_graph(result_1, result_2, result_3, result_4,x_lab,y_lab,tick_labels1,filename)

data = np.load('SPE_tr_ca.npy')
# data = pd.read_excel('spe_tr_ca.xlsx',header=None)
# data = np.array(data)
result_1, result_2, result_3, result_4 = data[0], data[1], data[2], data[3]
y_lab = 'Specificity'
x_lab = 'Training data(%)'
tick_labels1 = ['60', '70', '80', '90']
filename = 'SPE_tr_ca'
plot_graph(result_1, result_2, result_3, result_4,x_lab,y_lab,tick_labels1,filename)

###############################################################################################

data = np.load('ACC_kf_ca.npy')
# data = pd.read_excel('acc_kf_ca.xlsx',header=None)
# data = np.array(data)
result_1, result_2, result_3, result_4 = data[0], data[1], data[2], data[3]
y_lab = 'Accuracy'
x_lab = 'K-Fold'
tick_labels1 = ['5', '6', '7', '8']
filename = 'ACC_kf_ca'
plot_graph(result_1, result_2, result_3, result_4,x_lab,y_lab,tick_labels1,filename)

data = np.load('SEN_kf_ca.npy')
# data = pd.read_excel('sen_kf_ca.xlsx',header=None)
# data = np.array(data)
result_1, result_2, result_3, result_4 = data[0], data[1], data[2], data[3]
y_lab = 'Sensitivity'
x_lab = 'K-Fold'
tick_labels1 = ['5', '6', '7', '8']
filename = 'Sen_kf_ca'
plot_graph(result_1, result_2, result_3, result_4,x_lab,y_lab,tick_labels1,filename)

data = np.load('SPE_kf_ca.npy')
# data = pd.read_excel('spe_kf_ca.xlsx',header=None)
# data = np.array(data)
result_1, result_2, result_3, result_4 = data[0], data[1], data[2], data[3]
y_lab = 'Specificity'
x_lab = 'K-Fold'
tick_labels1 = ['5', '6', '7', '8']
filename = 'SPE_kf_ca'
plot_graph(result_1, result_2, result_3, result_4,x_lab,y_lab,tick_labels1,filename)