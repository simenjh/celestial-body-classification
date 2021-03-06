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






def plot_results_bayes(best_result):
    hyper_params = best_result["hyper_params"]
    cleaned_result = {"Network": [hyper_params["hidden_layers"]["network"]], "Batch size": best_result["batch_size"], "Learning rate": round(hyper_params["learning_rate"], 4), "Epochs": hyper_params["epochs"], "Regularization parameter": round(hyper_params["reg_param"], 4)}

    df = pd.DataFrame.from_dict(cleaned_result)
    table = plt.table(cellText=df.values, colLabels=df.columns, colWidths = [0.2]*len(df.columns), cellLoc = 'center', rowLoc = 'center', loc='top')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    plt.axis('off')
    plt.show()
