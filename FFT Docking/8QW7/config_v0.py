import ml_collections


config = {

    'resID_pocket_receptor': [65, 66, 67, 68, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],

    'resID_pocket_ligand': [12, 13, 60, 61, 62, 63, 64, 65, 66, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102],

    'dist_pocket_thredshold': 20,
    'angle_pocket_thredshold': 120,
    'fft_thredshold': -1000,

    'surface_intensity': 1,
    'pocket_intensity': 10,
    'protein_inner_intensity': -50,
    'ligand_inner_intensity': 1,

}

config = ml_collections.ConfigDict(config)

# receptor
resID_pocket_E3 = [173, 174, 175, 176, 177, 178, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220]
# ligand
resID_pocket_POI = [9, 10, 57, 58, 59, 60, 61, 62, 63, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

