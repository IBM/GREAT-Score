# This code is for producing figure 3 .

import matplotlib.pyplot as plt




Correlation_Coefficient = [0.5245098039215687, 0.6029411764705883, 0.6495098039215687,0.6593137254901962,0.661764705882353,0.661764705882353]  # Correlation Coefficient of action
Inception_Score= [6.55,6.61,8.64,8.83,9.46,10.43]  
labels = ['LSGAN','GGAN', 'SAGAN', 'SNGAN','DDPM','StyleGAN2']


plt.rcParams['axes.labelsize'] = 18  # 
plt.rcParams['xtick.labelsize'] = 14  # 
plt.rcParams['ytick.labelsize'] = 16  # 



width = 0.3  
x1_list = []
x2_list = []
for i in range(len(Correlation_Coefficient)):
    x1_list.append(i)
    x2_list.append(i + width)

fig, ax1 = plt.subplots(figsize=(10,6))


ax1.set_ylabel('Correlation Coefficient',color='lightseagreen')

ax1.set_ylim(0, 1)

ax1.bar(x1_list, Correlation_Coefficient, width=width, color='lightseagreen', align='edge')


ax1.set_xticklabels(ax1.get_xticklabels())  


ax2 = ax1.twinx()
ax2.set_ylabel('Inception Score',color='tab:blue')
ax2.set_ylim(5, 11)
ax2.bar(x2_list, Inception_Score, width=width, color='tab:blue', align='edge', tick_label=labels)

plt.tight_layout()
plt.show()
plt.savefig("Correlation Coefficient.png")