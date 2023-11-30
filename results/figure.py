import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# task id, minispec_token_count/minispec_token_count_std, minispec_planning_time/minispec_planning_time_std, 
result = """
1 & 7/0 & 1.1/0.2 & 43/0 & 4.7/0.4 \\
2 & 5/0 & 0.8/0.2 & 19.8/1.2 & 2.2/0.2 \\
3 & 11/0 & 1.5/0.4 & 16/0 & 1.6/0.2 \\
4 & 15/0 & 1.9/0.4 & 26/0 & 2.3/0.3 \\
5 & 22.2/0.1 & 2.1/0.3 & 65.2/0.7 & 5.9/1.1 \\
6 & 28/0 & 2.9/0.5 & 73/0 & 6.8/1.3 \\
7 & 23/0 & 2.5/0.3 & 66/0 & 6.5/0.7 \\
8 & 16/0 & 2.3/0.4 & 41/0 & 4.0/0.3 \\
9 & 11.8/0.5 & 1.5/0.3 & 26.4/0.2 & 3.3/0.5 \\
10 & 25.2/0.4 & 2.8/0.5 & 40/0 & 3.1/0.6 \\
11 & 62/0 & 5.4/0.7 & 129/0 & 13.4/1.8 \\
12 & 35/0 & 2.7/0.4 & 69/0 & 6.4/0.9"""
# calculate the average
result = result.split('\\')
result = [line.split('&') for line in result]
n_groups = 12
n_bars = 4

values = []
stds = []
for line in result:
    minispec_token_count = line[1].split('/')
    minispec_planning_time = line[2].split('/')
    python_token_count = line[3].split('/')
    python_planning_time = line[4].split('/')
    values.append([float(minispec_token_count[0]), float(python_token_count[0]), float(minispec_planning_time[0]), float(python_planning_time[0])])
    stds.append([float(minispec_token_count[1]), float(python_token_count[1]), float(minispec_planning_time[1]), float(python_planning_time[1])])

values = np.array(values)
stds = np.array(stds)

# Setting the positions and width for the bars
colors = ['#5877a2', '#85c5b1', '#e59344', '#f15f5f']
bar_width = 0.2
index = np.arange(n_groups)

plt.rcParams.update({'font.size': 14})
fig1, ax1 = plt.subplots(figsize=(8, 3))
fig2, ax2 = plt.subplots(figsize=(8, 3))

# Plotting bars 1 and 2 using the primary y-axis
ax1.bar(index + 0 * bar_width, values[:, 0], bar_width, yerr=stds[:, 0], label=f'MiniSpec Token Count', color=colors[1], capsize=2)
ax1.bar(index + 1 * bar_width, values[:, 1], bar_width, yerr=stds[:, 1], label=f'Python Token Count', color=colors[3], capsize=2)

# Plotting bars 3 and 4 using the secondary y-axis
ax2.bar(index + 0 * bar_width, values[:, 2], bar_width, yerr=stds[:, 2], label=f'MiniSpec Planning Time', color=colors[1], capsize=2)
ax2.bar(index + 1 * bar_width, values[:, 3], bar_width, yerr=stds[:, 3], label=f'Python Planning Time', color=colors[3], capsize=2)

# Labels, title, and legend
ax1.set_xlabel('Task ID')
ax1.set_ylabel('Token Count')
ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))
ax2.set_xlabel('Task ID')
ax2.set_ylabel('Planning Time (s)')
ax2.yaxis.set_major_locator(MaxNLocator(nbins=8))
# plt.title('Grouped Bar Chart with Two Scales')
ax1.set_xticks(index)
ax1.set_xticklabels([f'{i+1}' for i in range(n_groups)])
ax2.set_xticks(index)
ax2.set_xticklabels([f'{i+1}' for i in range(n_groups)])
fig1.subplots_adjust(top=0.95, bottom=0.2)
fig2.subplots_adjust(top=0.95, bottom=0.2)
ax1.legend(loc=(0.02, 0.7), fontsize=12)
ax2.legend(loc=(0.02, 0.7), fontsize=12)

# plt.show()
fig1.savefig('minispec_python_compare_token_count.pdf')
fig2.savefig('minispec_python_compare_planning_time.pdf')
