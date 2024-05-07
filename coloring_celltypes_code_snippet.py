from networkx import clustering


path_to_dicts = ""

phen_paths = {
    "IF1": path_to_dicts + "/IF1_phen_to_cell_mapping.csv",
    "IF2": path_to_dicts + "/IF2_phen_to_cell_mapping.csv",
    "IF3": path_to_dicts + "/IF3_phen_to_cell_mapping.csv",
}
phen_dict = clustering.phen_to_cell_dict(phen_paths["IF1"])

def rgb_to_hex(rgb):
    return '#' + '%02x%02x%02x' % rgb

IF1_cell_mapping = {"other": rgb_to_hex((190, 190, 190)), 
                    "CD15+Tumor": rgb_to_hex((73, 176, 248)),
                    "CD15-Tumor": rgb_to_hex((138, 79, 45)),
                    "Tcell": rgb_to_hex((235, 74, 148)),
                    "Bcell": rgb_to_hex((204, 49, 31)),
                    "BnTcell": rgb_to_hex((236, 95, 42)),
                    "Neutrophil": rgb_to_hex((0, 40, 245)),
                    "Macrophage": rgb_to_hex((97, 209, 62)),
                    "DC": rgb_to_hex((49, 113, 30))}

cell_dict = {
    "IF1": IF1_cell_mapping,
    "IF2": None, 
    "IF3": None,
}

def phenotype_to_color(ph, panel):
    return cell_dict[panel][phen_dict[ph]]
