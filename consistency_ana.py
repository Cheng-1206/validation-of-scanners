from scipy import stats
import xlrd


ori_excel_path = "./pair_data/extracted_measurement/mannequin consistency.xls"

ori_excel_content = xlrd.open_workbook(ori_excel_path)
ori_sheets = list(ori_excel_content.sheet_names())
excel_content = ori_excel_content.sheet_by_name(ori_sheets[0])

poly = []
fit3d = []

for row in range(1, 16):
    col_poly = 1  # 1 r-calf; 2 hip; 3 waist; 4 r-wrist
    col_fit3d = col_poly + 6
    poly.append(excel_content.cell_value(row, col_poly))
    fit3d.append(excel_content.cell_value(row, col_fit3d))


# Welch’s t-test
t_stat, p_val_t = stats.ttest_ind(poly, fit3d, equal_var=False)

# Mann–Whitney U test
u_stat, p_val_u = stats.mannwhitneyu(poly, fit3d, alternative='two-sided')

print("Welch t-test p:", p_val_t)
print("Mann-Whitney p:", p_val_u)