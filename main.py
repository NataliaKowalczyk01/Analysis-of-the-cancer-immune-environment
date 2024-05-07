import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.neighbors import radius_neighbors_graph
import numpy as np
import networkx as nx
import standardize as se
from scipy.sparse import find



# Funkcja do tworzenia grafu sąsiedztwa, dla konkretnego typu komórek
def create_graph(dataframe, radius, celltype):
    df = dataframe[dataframe["celltype"] == celltype]
    coordinates_list = df[['nucleus.x', 'nucleus.y']].values
    print("lista coordynatów")
    print(coordinates_list)
    graph = radius_neighbors_graph(coordinates_list, radius, mode='connectivity', metric='minkowski', p=2, metric_params=None, include_self=False)

    # Tworzenie słownika z indeksami i id komórek
    cell_IDs = df['cell.ID'].tolist() #lista indeksór komórek o okreslonym celltype
    index_to_cellID = {idx: cell_ID for idx, cell_ID in enumerate(cell_IDs)}
    
    return (graph.toarray(), index_to_cellID)

# Funkcja do tworzenia grafu sąsiedztwa, dla wszystkich typów komórek
def create_graph_all_cells(df, radius):
    coordinates_list = df[['nucleus.x', 'nucleus.y']].values
    graph = radius_neighbors_graph(coordinates_list, radius, mode='connectivity', metric='minkowski', p=2, metric_params=None, include_self=False)

    cell_IDs = df['cell.ID'].tolist() 
    cell_type = df['celltype'].tolist() 

    return (graph.toarray(), cell_IDs, cell_type)


def extract_scc_above_threshold(graph_array, threshold=15):
    # Konwersja macierzy sąsiedztwa na graf NetworkX
    G = nx.from_numpy_array(graph_array)
    # Ekstrakcja silnie spójnych składowych
    scc = list(nx.connected_components(G))
    scc_above_threshold = [comp for comp in scc if len(comp) > threshold]
    return scc_above_threshold

def map_numbers(components, mapping_dict):
    mapped_components = []
    for component in components:
        mapped_component = [mapping_dict[num] for num in component]
        mapped_components.append(mapped_component)
    return mapped_components

#funkcja zamieniająca krotki na same typy komórek znajdujących się w jednym komponencie
def convert_tuples_to_names(data):
    converted_data = {}
    for key, tuples_list in data.items():
        names_list = [item[1] for item in tuples_list]
        converted_data[key] = names_list
    return converted_data

#funkcja tworząca histogramy
def save_cell_type_histograms(data_dict, output_folder):
    for key, values in data_dict.items():
        # Liczymy wystąpienia każdego typu komórki
        cell_types_counts = {cell_type: values.count(cell_type) for cell_type in set(values)}
        
        # Tworzymy histogram
        plt.bar(cell_types_counts.keys(), cell_types_counts.values())
        plt.xlabel('Typ komórki')
        plt.ylabel('Liczba wystąpień')
        plt.title(f'Histogram typów komórek dla klucza {key}')
        
        # Zapisujemy histogram do pliku
        plt.savefig(f'{output_folder}/histogram_{key}.png')
        plt.close()  # Zamykamy obecny wykres, aby nie wyświetla



def main():
    df = pd.read_csv("./if_data/0197_IF1.csv")

    df_mapping = pd.read_csv("./IF1_phen_to_cell_mapping.csv")
    df['phenotype'] = df['phenotype'].apply(se.standardize_phenotype)
    merged_df = pd.merge(df, df_mapping, on='phenotype', how='left')

    # Wykres punktowy danych komórkowych
    fig, ax = plt.subplots()
    sns.scatterplot(x="nucleus.x", y="nucleus.y", data=merged_df, hue="celltype")
    plt.title('Scatterplot of cell data')
    plt.xlabel("x_column")
    plt.ylabel("y_column")
    plt.legend(title="Cell type")
    plt.grid(True)
    plt.savefig('scatterplot.png')
    plt.close()

    # Graf sąsiedztwa dla wszystkich komórek i dla komórek B
    graph_all, cell_ids, cell_type = create_graph_all_cells(merged_df, 30)
    graph_B, mapping_B  = create_graph(merged_df, 30, "Bcell")

    above_15_B = extract_scc_above_threshold(graph_B)
    ids_list_B = map_numbers(above_15_B, mapping_B)


    #Teraz dla kazdego komponentu z grafu tworzymy jego liste otoczenia
    component={}
    comp_number=0

    for comp in ids_list_B:
        comp_number+=1
        component[comp_number] = []
        for cell in comp:
            row_number=cell_ids.index(cell)
            neighbors= np.where(graph_all[row_number]!=0)[0] # to są indeksy wierszy -> znamy pozycje w macierzy
            neighbors_ids=[(cell_ids[index], cell_type[index]) for index in neighbors]
            component[comp_number]+=neighbors_ids
        neighbors_ids= list(set(neighbors_ids))

    converted_data = convert_tuples_to_names(component)
    save_cell_type_histograms(converted_data, 'output')









if __name__ == "__main__":
    main()
