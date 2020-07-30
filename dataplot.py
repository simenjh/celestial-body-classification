import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

    
def plot_class_distribution(data_file):
    df = pd.read_csv(data_file)
    temp_class_values = df["class"].values
    
    stars_count= len(np.where(temp_class_values == "STAR")[0])
    galaxies_count = len(np.where(temp_class_values == "GALAXY")[0])
    quasars_count = len(np.where(temp_class_values == "QSO")[0])
    celestial_bodies = ("STAR", "GALAXY", "QUASAR")
    celestial_body_counts = [stars_count, galaxies_count, quasars_count]
    
    y_pos = np.arange(len(celestial_bodies))

    bar = plt.bar(y_pos, celestial_body_counts, align='center', alpha=0.8)
    bar[0].set_color("r")
    bar[1].set_color("b")
    bar[2].set_color("g")
    plt.xticks(y_pos, celestial_bodies)
    plt.ylabel('Body count')
    plt.title('Celestial body distribution')

    plt.show()
