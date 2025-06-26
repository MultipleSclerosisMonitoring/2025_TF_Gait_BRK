import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def plot_2d(df, time_col, x_col, y_col=None, title="2D Time Series Plot"):
    """
    Plot a 2D time series graph, with one or two metrics plotted against time.

    :param df: DataFrame containing the data to plot.
    :type df: pd.DataFrame
    :param time_col: Column name representing the time values.
    :type time_col: str
    :param x_col: Column name representing the primary metric to be plotted on the x-axis.
    :type x_col: str
    :param y_col: (Optional) Column name representing the secondary metric to be plotted on the y-axis.
    :type y_col: str, optional
    :param title: (Optional) Title of the plot (default: "2D Time Series Plot").
    :type title: str, optional
    :return: None
    :rtype: None
    """
    if time_col not in df.columns or x_col not in df.columns or (y_col and y_col not in df.columns):
        raise ValueError("One or more specified columns are not in the DataFrame.")

    df[time_col] = pd.to_datetime(df[time_col])

    plt.figure(figsize=(12, 6))
    plt.plot(df[time_col], df[x_col], label=x_col, marker='o', linestyle='-')

    if y_col:
        plt.plot(df[time_col], df[y_col], label=y_col, marker='x', linestyle='-')

    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_3d(df, time_col, x_col, y_col, z_col, title="3D Visualization"):
    """
    Plot a 3D scatter plot, with three metrics plotted against each other and colored by time.

    :param df: DataFrame containing the data to plot.
    :type df: pd.DataFrame
    :param time_col: Column name representing the time values.
    :type time_col: str
    :param x_col: Column name representing the metric to be plotted on the x-axis.
    :type x_col: str
    :param y_col: Column name representing the metric to be plotted on the y-axis.
    :type y_col: str
    :param z_col: Column name representing the metric to be plotted on the z-axis.
    :type z_col: str
    :param title: (Optional) Title of the plot (default: "3D Visualization").
    :type title: str, optional
    :return: None
    :rtype: None
    """
    if any(col not in df.columns for col in [time_col, x_col, y_col, z_col]):
        raise ValueError("One or more specified columns are not in the DataFrame.")

    df[time_col] = pd.to_datetime(df[time_col])

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df[x_col], df[y_col], df[z_col], c=df[time_col].astype('int64'), cmap='viridis', alpha=0.8)
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(title)
    
    plt.show()

def plot_4d(df, time_col, x_col, y_col, z_col, title="4D Visualization"):
    """
    Plot a 3D scatter plot with time represented by color, effectively creating a 4D visualization.

    :param df: DataFrame containing the data to plot.
    :type df: pd.DataFrame
    :param time_col: Column name representing the time values.
    :type time_col: str
    :param x_col: Column name representing the metric to be plotted on the x-axis.
    :type x_col: str
    :param y_col: Column name representing the metric to be plotted on the y-axis.
    :type y_col: str
    :param z_col: Column name representing the metric to be plotted on the z-axis.
    :type z_col: str
    :param title: (Optional) Title of the plot (default: "4D Visualization").
    :type title: str, optional
    :return: None
    :rtype: None
    """
    if any(col not in df.columns for col in [time_col, x_col, y_col, z_col]):
        raise ValueError("One or more specified columns are not in the DataFrame.")

    df[time_col] = pd.to_datetime(df[time_col])
    time_numeric = (df[time_col] - df[time_col].min()).dt.total_seconds()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(df[x_col], df[y_col], df[z_col], c=time_numeric, cmap='viridis', alpha=0.8)
    plt.colorbar(sc, label="Time (seconds since start)")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(title)

    plt.show()

def plot_dual_3d(df, time_col, group1, group2, title="Dual 3D Plot"):
    """
    Plot two 3D scatter plots side-by-side with different sets of metrics, each plotted against time.

    :param df: DataFrame containing the data to plot.
    :type df: pd.DataFrame
    :param time_col: Column name representing the time values.
    :type time_col: str
    :param group1: List of two columns representing the first set of metrics to plot in the first subplot.
    :type group1: list[str]
    :param group2: List of two columns representing the second set of metrics to plot in the second subplot.
    :type group2: list[str]
    :param title: (Optional) Title of the plot (default: "Dual 3D Plot").
    :type title: str, optional
    :return: None
    :rtype: None
    """
    fig = plt.figure(figsize=(16, 8))

    for i, (group, subplot) in enumerate(zip([group1, group2], [121, 122])):
        ax = fig.add_subplot(subplot, projection='3d')
        ax.scatter(df[group[0]], df[group[1]], df[time_col], c=df[time_col].astype('int64'), cmap='coolwarm', alpha=0.8)
        ax.set_xlabel(group[0])
        ax.set_ylabel(group[1])
        ax.set_zlabel("Time")
        ax.set_title(f"{group[0]} vs {group[1]}")

    plt.show()
