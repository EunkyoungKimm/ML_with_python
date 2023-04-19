import numpy as np
names = ['DFF R-FCN', 'R-FCN', 'FGFA R-FCN']

dff_data = np.array([(0.581, 13.5),(0.598, 12.8),(0.618, 11.7),
           (0.62, 11.3), (0.624, 10.2), (0.627, 9.8),
           (0.629, 9.2), (0.63, 9)])
r_data = np.array([(0.565, 11.2), (0.645, 9)])
fgfa_data = np.array([(0.63, 8.8), (0.653, 9.3), (0.664, 9.6), (0.676, 10.1)])

dff_text = ['1:20', '1:15', '1:10', '1:8','1:5', '1:3', '1:2', '1:1']
r_text = ['Half Model', 'Full Model']
fgfa_text = ['1:1', '3:1', '7:1', '19:1']

data_list = [dff_data, r_data, fgfa_data]
text_list = [dff_text, r_text, fgfa_text]

print(data_list)

a = zip(data_list, text_list)
print(a)
for data_idx, (data, text_arr) in enumerate(zip(data_list, text_list)):
    print(data_idx, (data, text_arr))
    print(data)
    print(text_arr)
