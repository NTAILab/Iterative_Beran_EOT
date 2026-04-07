import numpy as np
import matplotlib.pyplot as plt
import torch

def save_results(model_baseline, model_new, baseline_history, train_history, test_input, test_deltas, test_times):
    with torch.no_grad():
        times, SF = model_new.survival_curve(test_input)
        times, SF_baseline = model_baseline.survival_curve(test_input)

        for k in train_history.keys():
            plt.figure(figsize=(11, 6))
            if len(train_history[k][0]) > 0:
                plt.plot(range(len(train_history[k][1])), train_history[k][1], linewidth=2.0, linestyle='-',
                         label=f'model', color='blue', marker='o')
            else:
                plt.plot(range(len(train_history[k][1])), train_history[k][1], linewidth=2.0, linestyle='-',
                         label=f'{k}', color='blue', marker='o')

            plt.xlabel('iterations')
            plt.ylabel(f'{k}')
            plt.title(f'{k}')
            plt.grid(True, alpha=0.3)
            if k in baseline_history.keys():
                plt.plot([baseline_history[k]]*len(train_history[k][1]),
                            color='red',
                            linestyle='--',
                            linewidth=2.0,
                            marker='s',
                            alpha=0.85,
                            label=f'baseline')
            else:
                plt.axhline(y=train_history[k][1][0],
                            color='blue',
                            linestyle='-.',
                            linewidth=2.0,
                            alpha=0.85,
                            label=f'start {k}')
            plt.legend()
            plt.show()

        for i in range(test_input.shape[0]):
            plt.figure(figsize=(11, 6))
            plt.step(times.tolist() + [times[-1] + 0.00001], SF[i].tolist() + [0], where='post', linewidth=2.0, label=f'S(t | x)', color='blue')
            plt.step(times.tolist() + [times[-1] + 0.00001], SF_baseline[i].tolist() + [0], where='post', linewidth=1.0, label=f'S(t | x) (baseline)', color='red')

            plt.xlabel('t')
            plt.ylabel('S(t | x)')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.05)
            plt.axvline(x=test_times[i],
                        color='black',
                        linestyle='--',
                        linewidth=2.0,
                        alpha=0.85,
                        label=f'time of event')

            integrated_value_b = model_baseline.integrate_SF(SF_baseline[np.newaxis, i])
            integrated_value = model_new.integrate_SF(SF[np.newaxis, i])

            plt.axvline(x=integrated_value_b,
                        color='red',
                        linestyle='-.',
                        linewidth=1.0,
                        alpha=0.85,
                        label=f'mean value (baseline)')
            plt.axvline(x=integrated_value,
                        color='blue',
                        linestyle='-.',
                        linewidth=2.0,
                        alpha=0.85,
                        label=f'mean value')
            plt.legend()
            plt.show()