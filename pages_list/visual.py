from time import time
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st

# For plotting
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import plotly.graph_objects as go

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

#For standardising the dat
from sklearn.preprocessing import StandardScaler

#PCA
from sklearn.manifold import TSNE

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)


# Encoding all the images for inclusion in a dataframe.

from io import BytesIO
from PIL import Image
import base64


def embeddable_image(data):
    img_data = 255 - 15 * data.astype(np.uint8)
    image = Image.fromarray(img_data, mode='L').resize((28,28), Image.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()

# loading up bokeh and other tools to generate a suitable interactive plot.

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10

output_notebook()


    # def plot_digits(data):
    #     fig, axes = plt.subplots(4, 10, figsize=(10, 4),
    #                             subplot_kw={'xticks':[], 'yticks':[]},
    #                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    #     for i, ax in enumerate(axes.flat):
    #         ax.imshow(data[i].reshape(28, 28),
    #                 cmap='binary', interpolation='nearest',
    #                 clim=(0, 16))
    #     st.pyplot(fig)  # Отображение рисунка в Streamlit

    # with open('data/tsne.npy', 'rb') as f:
    #     tsne = np.load(f)


    # def plot_tsne(tsne):
        
    #     plt.figure(figsize=(8, 6))
    #     plt.scatter(tsne[:, 0], tsne[:, 1], s=5, c=y_subset, cmap='Spectral')
    #     plt.gca().set_aspect('equal', 'datalim')
    #     plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    #     plt.title('Visualizing Kannada MNIST through t-SNE', fontsize=24)
    #     st.pyplot()

    # plot_tsne(tsne)

    # with open('data/pca_tsne.npy', 'rb') as f:
    #     pca_tsne = np.load(f)

    # def plot_tsne_pca(pca_tsne):

    #     plt.scatter(pca_tsne[:, 0], pca_tsne[:, 1], s= 5, c=y_subset, cmap='Spectral')
    #     plt.gca().set_aspect('equal', 'datalim')
    #     plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    #     plt.title('Visualizing Kannada MNIST through t-SNE with PCA', fontsize=24)


    # plot_tsne_pca(pca_tsne)


def visual():
    with open('data/pca_tsne2.npy', 'rb') as f:
        pca_tsne2 = np.load(f)

    with open('data/pca_tsne.npy', 'rb') as f:
        pca_tsne = np.load(f)
    x=pca_tsne2[:, 0]
    y=pca_tsne2[:, 1]
    z=pca_tsne2[:, 2]

    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=12,
            color=x,                # set color to an array/list of desired values
            colorscale='Spectral',   # choose a colorscale
            opacity=0.8
        )
    )])

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    # Отображение интерактивного графика в Streamlit
    is_button_clicked = False

    # Создание кнопки
    button_label = "Показать интерактивный график" if not is_button_clicked else "Закрыть"
    if st.button(button_label):
        # Изменение состояния при клике
        is_button_clicked = not is_button_clicked
        st.plotly_chart(fig)


    train = pd.read_csv('data/train.csv')

    y = train.loc[:,'label'].values
    x = train.loc[:,'pixel0':].values

    x_subset = x[0:10000]
    y_subset = y[0:10000]


    x_subset_reshape = x_subset.reshape(-1,28,28)

    digits_df = pd.DataFrame(pca_tsne, columns=('x', 'y'))
    digits_df['digit'] = [str(x) for x in y_subset]
    digits_df['image'] = list(map(embeddable_image, x_subset_reshape))


    datasource = ColumnDataSource(digits_df)
    color_mapping = CategoricalColorMapper(factors=[str(9 - x) for x in y_subset],
                                        palette=Spectral10)

    plot_figure = figure(
    title='t-SNE projection of the Kannada MNIST dataset',
    width=600,
    height=600,
    tools=('pan, wheel_zoom, reset')
)

    # Добавление всплывающей подсказки
    hover = HoverTool(tooltips="""
        <div>
            <div>
                <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
            </div>
            <div>
                <span style='font-size: 16px; color: #224499'>Digit:</span>
                <span style='font-size: 18px'>@digit</span>
            </div>
        </div>
    """)
    plot_figure.add_tools(hover)

    # Создание кругов на графике
    plot_figure.circle(
        'x',
        'y',
        source=datasource,
        color=dict(field='digit', transform=color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4
    )

    # Отображение графика на Streamlit
    st.bokeh_chart(plot_figure)