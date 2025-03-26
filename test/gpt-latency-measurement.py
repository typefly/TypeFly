import time
import tiktoken

import sys
sys.path.append('..')
from typefly.llm_wrapper import LLMWrapper

llm = LLMWrapper()

def output_measure_prompt(length):
    prompt = 'Please generate the exact same output as the following text: '
    for i in range(length // 2):
        prompt += str(i % 10) + " "
    return prompt

def input_measure_prompt(length, model_name):
    suffix = "Please ignore all the above text and just generate True"
    prompt = ''
    enc = tiktoken.encoding_for_model(model_name)
    init_len = enc.encode(suffix)
    for i in range((length - len(init_len)) // 2):
        prompt += str(i % 10) + " "
    return prompt + suffix

def measure(model_name="gpt-4o", input: bool = False):
    lengths = [50, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000] if input else [50, 100, 200, 300, 400]
    result = []
    enc = tiktoken.encoding_for_model(model_name)
    for length in lengths:
        accu_t = 0
        accu_input_length = 0
        accu_output_length = 0
        for i in range(10):
            prompt = input_measure_prompt(length, model_name) if input else output_measure_prompt(length)
            start = time.time()
            output = llm.request(prompt, model_name)
            dt = time.time() - start

            input_length = len(enc.encode(prompt))
            output_length = len(enc.encode(output))

            accu_t += dt
            accu_input_length += input_length
            accu_output_length += output_length

            print(f"dt: {dt}, i_len: {input_length}, o_len: {output_length}")
        
        print("Time taken for length", length, ":", accu_t / 10)
        print("Input length:", accu_input_length / 10)
        print("Output length:", accu_output_length / 10)
        result.append((length, accu_t / 10, accu_input_length / 10, accu_output_length / 10))
    print(result)

###### measurement
# measure("gpt-4o", False)
# exit()

measured_data = [
    {
        "date": "2024-03-05",
        "model": "gpt-4",
        "input": [
            (50, 0.5378251791000366, 49.0, 1.0),
            (500, 0.5108302307128907, 499.0, 1.0),
            (1000, 0.4951801300048828, 999.0, 1.0),
            (2000, 0.5111032485961914, 1999.0, 1.0),
            (3000, 0.5264493227005005, 2999.0, 1.0),
            (4000, 0.5382437705993652, 3999.0, 1.0),
            (5000, 0.5212562322616577, 4999.0, 1.0),
            (6000, 0.5919422626495361, 5999.0, 1.0),
            (7000, 0.5916801214218139, 6999.0, 1.0),
            (8000, 0.6088189125061035, 7999.0, 1.0)
        ],
        "output": [
            (50, 2.276, 62, 49),
            (100, 4.560, 112, 99),
            (200, 8.473, 212, 199),
            (300, 10.996, 312, 295),
            (400, 14.425, 412, 413)
        ]
    },
    {
        "date": "2025-03-10",
        "model": "gpt-4",
        "input": [
            (50, 0.6257302761077881, 49.0, 1.0),
            (500, 0.6683852434158325, 499.0, 1.0),
            (1000, 0.6935260772705079, 999.0, 1.0),
            (2000, 0.941708779335022, 1999.0, 1.0),
            (3000, 0.666994833946228, 2999.0, 1.0),
            (4000, 0.7132920980453491, 3999.0, 1.0),
            (5000, 0.8775372982025147, 4999.0, 1.0),
            (6000, 0.7193283081054688, 5999.0, 1.0),
            (7000, 0.8627294540405274, 6999.0, 1.0),
            (8000, 0.6080472707748413, 7999.0, 1.0)
        ],
        "output": [
            (50, 3.1781239748001098, 62.0, 49.0),
            (100, 5.728551363945007, 112.0, 99.0),
            (200, 10.729672980308532, 212.0, 195.0),
            (300, 15.626528573036193, 312.0, 293.0),
            (400, 18.929732251167298, 412.0, 413.0)
        ]
    },
    {
        "date": "2025-03-10",
        "model": "gpt-4o",
        "input": [
            (50, 0.38800146579742434, 49.0, 1.0),
            (500, 0.3816303968429565, 499.0, 1.0),
            (1000, 0.4146378993988037, 999.0, 1.0),
            (2000, 0.4167546510696411, 1999.0, 1.0),
            (3000, 0.46774892807006835, 2999.0, 1.0),
            (4000, 0.5402949333190918, 3999.0, 1.0),
            (5000, 0.5743841409683228, 4999.0, 1.0),
            (6000, 0.61163170337677, 5999.0, 1.0),
            (7000, 0.5890320539474487, 6999.0, 1.0),
            (8000, 0.6145241737365723, 7999.0, 1.0)
        ],
        "output": [
            (50, 1.555180263519287, 62.0, 49.0),
            (100, 3.0511394262313845, 112.0, 99.3),
            (200, 5.575378036499023, 212.0, 199.0),
            (300, 7.651046824455261, 312.0, 301.8),
            (400, 9.970797681808472, 412.0, 402.0)
        ]
    }
]

####### draw the plot
network_latency = 28.127

red_color = '#FF6B6B'
blue_color = '#4D96FF'
white_color = '#FFFFFF'
black_color = '#000000'

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

model_name = measured_data[2]["model"]
data_input_measure = measured_data[2]["input"]
data_output_measure = measured_data[2]["output"]

col1_1 = [x[0] for x in data_input_measure]
col2 = [x[1] for x in data_input_measure]

col1_2 = [x[0] for x in data_output_measure]
col3 = [x[1] for x in data_output_measure]

# Perform linear regression for each dataset
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(col1_1, col2)

for i in range(len(data_output_measure)):
    col3[i] -= data_output_measure[i][2] * slope2

slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(col1_2, col3)

# Create arrays from the x-coordinates for line plots
line_x1 = np.linspace(min(col1_1), max(col1_1), 100)  # For smoother line plot
line_x2 = np.linspace(min(col1_2), max(col1_2), 100)

# Create line equations for the plots
line2 = slope2 * line_x1 + intercept2
line3 = slope3 * line_x2 + intercept3

plt.rcParams.update({'legend.fontsize': 19, 'axes.edgecolor': 'black',
                     'axes.linewidth': 2.2, 'font.size': 25})

### plot in a single figure
# fig, ax1 = plt.subplots(figsize=[16, 6])
# plt.tight_layout(pad=2)
# # Plot the first dataset with its regression
# ax1.scatter(col1_1, col2, color=black_color, label='Various input, fixed output', marker='x', linewidth=3, s=200)
# ax1.plot(np.array(col1_1), slope2 * np.array(col1_1) + intercept2, '-', color=black_color, label=f'a={slope2:.6f}, r={r_value2:.4}', linewidth=3)
# ax1.set_xlabel('Input Token Number', color=black_color)
# ax1.set_ylabel('Time Taken (s)')
# ax1.tick_params(axis='x', labelcolor=black_color)
# ax1.tick_params(axis='y')

# # Create a second x-axis for the second dataset
# ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
# ax2.scatter(col1_2, col3, color=black_color, label='Fixed input, various output', linewidth=3, s=200)
# ax2.plot(np.array(col1_2), slope3 * np.array(col1_2) + intercept3, '--', color=black_color, label=f'b={slope3:.6f}, r={r_value3:.4}', linewidth=3)
# ax2.set_xlabel('Output Token Number', color=black_color)
# ax2.tick_params(axis='x', labelcolor=black_color)

# # Add legends and show plot
# ax1.legend(loc='lower right', bbox_to_anchor=(1, 0.12))
# ax2.legend(loc='upper left')
# # plt.show()

# plt.savefig('gpt4-latency.pdf')

### plot in two figures
fig, ax1 = plt.subplots(figsize=[14, 6])
plt.tight_layout(pad=2)
# Plot the first dataset with its regression
ax1.scatter(col1_1, col2, color=black_color, label='Various input, fixed output', marker='x', linewidth=3, s=200)
ax1.plot(np.array(col1_1), slope2 * np.array(col1_1) + intercept2, '-', color=black_color, label=f'a={slope2:.6f}, r={r_value2:.4}', linewidth=3)
ax1.set_xlabel('Input Token Number', color=black_color)
ax1.set_ylabel('Time Taken (s)')
ax1.tick_params(axis='x', labelcolor=black_color)
ax1.tick_params(axis='y')

# Add legends and show plot
ax1.legend(loc='upper left')
plt.savefig(model_name + '-latency-input.pdf')
# plt.show()

fig, ax2 = plt.subplots(figsize=[14, 6])
plt.tight_layout(pad=2)
# Create a second x-axis for the second dataset
ax2.scatter(col1_2, col3, color=black_color, label='Fixed input, various output', linewidth=3, s=200)
ax2.plot(np.array(col1_2), slope3 * np.array(col1_2) + intercept3, '--', color=black_color, label=f'b={slope3:.6f}, r={r_value3:.4}', linewidth=3)
ax2.set_xlabel('Output Token Number', color=black_color)
ax2.set_ylabel('Time Taken (s)')
ax2.tick_params(axis='x', labelcolor=black_color)
ax2.tick_params(axis='y')

# Add legends and show plot
ax2.legend(loc='upper left')
plt.savefig(model_name + '-latency-output.pdf')
# plt.show()
