import matplotlib.pyplot as plt
import numpy as np
import io

def generate_sine_plot():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Sine Wave")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_cosine_plot():
    x = np.linspace(0, 10, 100)
    y = np.cos(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, color='red')
    ax.set_title("Cosine Wave")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_scatter_plot():
    np.random.seed(0)
    x = np.random.rand(50)
    y = np.random.rand(50)

    fig, ax = plt.subplots()
    ax.scatter(x, y, color='green')
    ax.set_title("Random Scatter")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf