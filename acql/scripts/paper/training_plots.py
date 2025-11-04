import matplotlib.pyplot as plt

import mlflow
import pandas as pd
import seaborn as sns

from scipy.ndimage.filters import gaussian_filter1d


label_size = 'x-large'

spec_rename_map = {
    'SimpleMazeTwoSubgoals': 'PointMass - Sequence',
    'SimpleMazeBranching1': 'PointMass - Branch',
    'SimpleMazeObligationConstraint1': 'PointMass - Safety',
    'SimpleMazeUntil2': 'PointMass - Until',
    'SimpleMazeLoopWithObs': 'PointMass - Loop',
    'SimpleMaze3DTwoSubgoals': 'Quadcopter - Sequence',
    'SimpleMaze3DBranching1': 'Quadcopter - Branch',
    'SimpleMaze3DObligationConstraint2': 'Quadcopter - Safety',
    'SimpleMaze3DUntil2': 'Quadcopter - Until',
    'SimpleMaze3DLoopWithObs': 'Quadcopter - Loop',
    'AntMazeTwoSubgoals': 'AntMaze - Sequence',
    'AntMazeBranching1': 'AntMaze - Branch',
    'AntMazeObligationConstraint3': 'AntMaze - Safety',
    'AntMazeUntil1': 'AntMaze - Until',
    'AntMazeLoopWithObs': 'AntMaze - Loop',
}

alg_rename_map = {
    "CRM": "CRM",
    "CRM_RS": "CRM-RS",
    "ACQL": "ACQL",
}

def apply_smoothing(df, sigma=3):
    return df.apply(lambda col: gaussian_filter1d(col, sigma) if col.dtype == "float64" else col, axis=0)


def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_axes['x'].join(target, ax)
        if sharey:
            target._shared_axes['y'].join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1,:].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:,1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)


def plot_grouped_success_rate_metrics(df):
    grouped = df.groupby(['spec', 'alg', 'step'])
    summary = grouped.agg({'sr': ['mean', 'std'], 'reward': ['mean', 'std']}).reset_index()
    summary.columns = ['spec', 'alg', 'step', 'sr_mean', 'sr_std', 'reward_mean', 'reward_std']

    # unique_specs = summary['spec'].unique()
    fig, axes = plt.subplots(5, 3, figsize=(15, 25))

    set_share_axes(axes[:,:], sharex=True, sharey=True)

    axes = axes.flatten()

    chosen_specs = [
        'SimpleMazeTwoSubgoals',
        'SimpleMaze3DTwoSubgoals',
        'AntMazeTwoSubgoals',
        'SimpleMazeBranching1',
        'SimpleMaze3DBranching1',
        'AntMazeBranching1',
        'SimpleMazeObligationConstraint1',
        'SimpleMaze3DObligationConstraint2',
        'AntMazeObligationConstraint3',
        'SimpleMazeUntil2',
        'SimpleMaze3DUntil2',
        'AntMazeUntil1',
        'SimpleMazeLoopWithObs',
        'SimpleMaze3DLoopWithObs',
        'AntMazeLoopWithObs'
    ]

    for idx, spec in enumerate(chosen_specs):
        spec_df = summary[summary['spec'] == spec]
        name = spec_rename_map[spec]
        
        ax = axes[idx]

        for alg in spec_df['alg'].unique():
            alg_df = spec_df[spec_df['alg'] == alg]
            alg_name = alg_rename_map[alg]

            if alg == "HDCQN_AUTOMATON_HER":
                color = 'green'
            elif alg == "CRM_RS":
                color = 'blue'
            elif alg == "CRM":
                color = 'orange'
            else:
                color = None

            # alg_df = apply_smoothing(alg_df)

            sns.lineplot(x='step', y='sr_mean', data=alg_df, label=f'{alg_name}', ax=ax, color=color, legend=False)
            # ax.fill_between(alg_df['step'], alg_df['sr_mean'] - (alg_df['sr_std'] / math.sqrt(3)),
            #                 alg_df['sr_mean'] + (alg_df['sr_std'] / math.sqrt(3)), alpha=0.2, color=color)
            ax.fill_between(alg_df['step'], alg_df['sr_mean'] - (alg_df['sr_std']),
                            alg_df['sr_mean'] + (alg_df['sr_std']), alpha=0.2, color=color)

        ax.set_title(f'{name}')
        ax.set_xlabel('Timesteps', fontsize=label_size)
        ax.set_ylabel('Success Rate', fontsize=label_size)
        ax.grid()

    # Get handles and labels from one of the subplots (e.g., the last one)
    handles, labels = ax.get_legend_handles_labels()

    # Add a single legend to the figure
    fig.legend(handles, labels, fontsize='xx-large', loc='lower center', fancybox=True, shadow=True, ncol=2, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig("./training-plots-sr.pdf")
    # plt.show()


def plot_grouped_reward_metrics(df):
    grouped = df.groupby(['spec', 'alg', 'step'])
    summary = grouped.agg({'sr': ['mean', 'std'], 'reward': ['mean', 'std']}).reset_index()
    summary.columns = ['spec', 'alg', 'step', 'sr_mean', 'sr_std', 'reward_mean', 'reward_std']

    # unique_specs = summary['spec'].unique()
    fig, axes = plt.subplots(5, 3, figsize=(15, 25))

    set_share_axes(axes[:,:], sharex=True)
    set_share_axes(axes[:4,:], sharey=True)

    axes = axes.flatten()

    chosen_specs = [
        'SimpleMazeTwoSubgoals',
        'SimpleMaze3DTwoSubgoals',
        'AntMazeTwoSubgoals',
        'SimpleMazeBranching1',
        'SimpleMaze3DBranching1',
        'AntMazeBranching1',
        'SimpleMazeObligationConstraint1',
        'SimpleMaze3DObligationConstraint2',
        'AntMazeObligationConstraint3',
        'SimpleMazeUntil2',
        'SimpleMaze3DUntil2',
        'AntMazeUntil1',
        'SimpleMazeLoopWithObs',
        'SimpleMaze3DLoopWithObs',
        'AntMazeLoopWithObs'
    ]

    for idx, spec in enumerate(chosen_specs):
        spec_df = summary[summary['spec'] == spec]
        name = spec_rename_map[spec]

        ax = axes[idx]

        for alg in spec_df['alg'].unique():
            alg_df = spec_df[spec_df['alg'] == alg]
            alg_name = alg_rename_map[alg]

            if alg == "HDCQN_AUTOMATON_HER":
                color = 'green'
            elif alg == "CRM_RS":
                color = 'blue'
            elif alg == "CRM":
                color = 'orange'
            else:
                color = None

            # alg_df = apply_smoothing(alg_df)

            sns.lineplot(x='step', y='reward_mean', data=alg_df, label=f'{alg_name}', ax=ax, color=color, legend=False)
            # ax.fill_between(alg_df['step'], alg_df['reward_mean'] - alg_df['reward_std'] / math.sqrt(3),
            #                 alg_df['reward_mean'] + alg_df['reward_std'] / math.sqrt(3), alpha=0.2, color=color)
            ax.fill_between(alg_df['step'], alg_df['reward_mean'] - alg_df['reward_std'],
                            alg_df['reward_mean'] + alg_df['reward_std'], alpha=0.2, color=color)

        ax.set_title(f'{name}')
        ax.set_xlabel('Timesteps', fontsize=label_size)
        ax.set_ylabel('Average Reward', fontsize=label_size)
        ax.grid()

    # Get handles and labels from one of the subplots (e.g., the last one)
    handles, labels = ax.get_legend_handles_labels()

    # Add a single legend to the figure
    fig.legend(handles, labels, fontsize='xx-large', loc='lower center', fancybox=True, shadow=True, ncol=2, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig("./training-plots-reward.pdf")
    # plt.show()


if __name__ == "__main__":

    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")

    lab_pc_df = pd.read_csv("./lab-pc-data.csv")
    phd_server_df = pd.read_csv("./phd-server-data.csv")
    df = pd.concat([lab_pc_df, phd_server_df])
    df = df.set_index("Unnamed: 0")

    if not df.empty:
        plot_grouped_reward_metrics(df)
        plot_grouped_success_rate_metrics(df)
    else:
        print("No relevant runs found.")
