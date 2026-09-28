import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import xlrd
import numpy as np


ori_excel_path = "./pair_data/extracted_measurement/consistencydata/repeatmsr.xls"

ori_excel_content = xlrd.open_workbook(ori_excel_path)
ori_sheets = list(ori_excel_content.sheet_names())
excel_content = ori_excel_content.sheet_by_name(ori_sheets[0])

data_A = np.empty(15)  # tape
data_B = np.empty(15)  # PolyCam

for id in range(0, 15):

    data_A[id] = excel_content.cell_value(id+1, 7)  # 7 for calf; 8 for hip; 9 for waist; 10 for wrist
    # print(data_A[0, row])
    data_B[id] = excel_content.cell_value(id+1, 1)  # A - 6


data_A_list = data_A.tolist()
data_B_list = data_B.tolist()


fig, ax = plt.subplots(figsize=(7,5))


# Plot boxplots
positions_A = [1]
positions_B = [2]
ax.boxplot(data_A_list, positions=positions_A, widths=0.6)
ax.boxplot(data_B_list, positions=positions_B, widths=0.6)

# Plot Scatter
# x_list = np.ones(15)
# x_list_A = x_list.tolist()
# x_list_B = (x_list*2).tolist()
# ax.scatter(x_list, data_A_list)
# ax.scatter(x_list_B, data_B_list)

# X-axis labels (centered between pairs)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Tape', 'PolyCam'])


# Labels
ax.set_ylabel('Values (mm)', fontname='Times New Roman', fontsize=18)
ax.set_title('Tape vs PolyCam Measurements on Mannequin Right Calf', fontname='Times New Roman', fontsize=18)

ax.tick_params(axis='both', labelfontfamily='Times New Roman', labelsize=15)


plt.show()

