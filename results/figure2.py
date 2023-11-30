import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

data = {
    'Task_ID': [5, 6, 7, 9, 10, 11],
    'With_Query_Token_Count': [22, 27.6, 23.5, 11.4, 25.4, 61.6],
    'Without_Query_Token_Count': [21, 8, 16, 15, 18, 103.8],
    'With_Query_Token_Count_Std': [0, 0.5, 0.5, 0.8, 0.6, 1.8],
    'Without_Query_Token_Count_Std': [0, 0, 0, 0, 0, 2.9],
    'With_Query_Planning_Time': [20.8, 26.6, 15.9, 16.7, np.nan, 37.3],
    'Without_Query_Planning_Time': [np.nan, 14.0, np.nan, np.nan, np.nan, 279.3],
    'With_Query_Planning_Time_Std': [2.1, 1.9, 1.3, 1.6, np.nan, 3.1],
    'Without_Query_Planning_Time_Std': [np.nan, 1.8, np.nan, np.nan, np.nan, 10.8]
}


current_directory = os.path.dirname(os.path.abspath(__file__))

patterns = ['/', ' ']

red_color = '#FF6B6B'
blue_color = '#4D96FF'

plt.rcParams.update({'legend.fontsize': 27, 'axes.edgecolor': 'black',
                     'axes.linewidth': 3.0, 'font.size': 27})

# Create DataFrame
df = pd.DataFrame(data)

# Plot for Token Count Comparison
fig, ax = plt.subplots(figsize=(14, 5))
x_positions = np.arange(len(df['Task_ID']))

for i, task in enumerate(x_positions):
    ax.bar(task - 0.2, df['With_Query_Token_Count'][i], color=red_color, hatch=patterns[0], width=0.4,
           yerr=df['With_Query_Token_Count_Std'][i], capsize=5, error_kw=dict(capthick=2), edgecolor='black', linewidth=3, label='With Query' if i == 0 else "")
    ax.bar(task + 0.2, df['Without_Query_Token_Count'][i], color=blue_color, hatch=patterns[1], width=0.4,
           yerr=df['Without_Query_Token_Count_Std'][i], capsize=5, error_kw=dict(capthick=2), edgecolor='black', linewidth=3, label='Without Query' if i == 0 else "")

ax.set_xticks(x_positions)
ax.set_xticklabels(df['Task_ID'])
ax.set_xlabel('Task ID')
ax.set_ylabel('Token Count')
ax.legend()
plt.tight_layout()

# Save the Token Count comparison figure
token_count_fig_path = current_directory + '/query_token_count_comparison.svg'
fig.savefig(token_count_fig_path, format='svg', bbox_inches='tight')

# Plot for Execution Time Comparison
fig, ax = plt.subplots(figsize=(14, 5))

legend_added = {'With Query': False, 'Without Query': False}

for i, task in enumerate(x_positions):
    if not np.isnan(df['With_Query_Planning_Time'][i]):
        label = 'With Query' if not legend_added['With Query'] else ""
        ax.bar(task - 0.2, df['With_Query_Planning_Time'][i], color=red_color, hatch=patterns[0], width=0.4,
               yerr=df['With_Query_Planning_Time_Std'][i], capsize=5, error_kw=dict(capthick=2), edgecolor='black', linewidth=3, label=label)
        legend_added['With Query'] = True
    else:
        ax.plot(task - 0.2, 15, 'x', color='black', markersize=20, markeredgewidth=3)

    if not np.isnan(df['Without_Query_Planning_Time'][i]):
        label = 'Without Query' if not legend_added['Without Query'] else ""
        ax.bar(task + 0.2, df['Without_Query_Planning_Time'][i], color=blue_color, hatch=patterns[1], width=0.4,
               yerr=df['Without_Query_Planning_Time_Std'][i], capsize=5, error_kw=dict(capthick=2), edgecolor='black', linewidth=3, label=label)
        legend_added['Without Query'] = True
    else:
        ax.plot(task + 0.2, 15, 'x', color='black', markersize=20, markeredgewidth=3)


ax.set_xticks(x_positions)
ax.set_xticklabels(df['Task_ID'])
ax.set_xlabel('Task ID')
ax.set_ylabel('Execution Time (s)')
ax.legend()
plt.tight_layout()

# Save the Planning Time comparison figure
planning_time_fig_path = current_directory + '/query_planning_time_comparison.svg'
fig.savefig(planning_time_fig_path, format='svg', bbox_inches='tight')


