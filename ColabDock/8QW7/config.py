import ml_collections
import joblib


config = {
    # path where you want to save the results
    'save_path': './PROTAC_8QW7_results/FFT_cluster_X',

    ###########################################################################################################
    # template and native structure
    ###########################################################################################################
    # the structure of proteins you want to dock
    'template': './PROTAC_8QW7_results/FFT_pocket/POI_and_E3_cluster_X.pdb',

    # optional, the native structure of the complex, used for calculating the RMSD
    # if you do not have native structure, set it to None.
    #'native': './protein/PROTAC_8QW7/8QW7_AB.pdb',
    'native': None,

    # docking chains
    # This determines the order that the chain sequences are concatenated to form the complex sequence.
    'chains': 'A,B',

    # input the chainIDs if you want the relative position of chains is fixed as in the provided template
    # otherwise, set to None
    # example:
    #     'fixed_chains': ['A,B', 'C,D']
    #     the relative position of chain A and B is fixed, also that of chain C and D.
    'fixed_chains': None,
    
    ###########################################################################################################
    # experimental restraints
    ###########################################################################################################
    # the threshold of the experimental restraints, usually set to 8.0Å.
    # Change to other values if you know the threshold of the restraints you provide.
    # Due to the definition of distogram in AF2, threshold should be set to a value between 2Å and 22Å
    'res_thres': 8.0,

    # 1v1 restraints
    # description:
    #     The distance between two residues is below a given threshold (res_thres).
    #     If there is no such restraints, set to None.
    #     If you have multiple 1v1 restraints, list them in a [].
    #     The order number in a 1v1 restraint refers to the residue in the complex sequence.
    #         The complex sequence is concatenated by the chain sequences and the order is determined by the "chains" provided above.
    #         This is the same for the remaining types of restraints.
    #     The order number starts from 1.
    # example:
    #     'rest_1v1': [[78,198],[20,50]]
    #     The distance between 78th and 198th residue is below a given threshold, as well as the distance between 20th and 50th residue.
    'rest_1v1': None,

    # 1vN restraints
    # description:
    #     The distance between one residue and a residue set is below a given threshold (res_thres).
    #     If there is no such restraints, set to None.
    #     If you have multiple 1v1 restraints, list them in [].
    #     The order number starts from 1.
    # example:
    #     'rest_1vN': [36,list(range(160,171))+[178,190]]
    #     The distance between the 36th residue and at least a residue from 160th to 170th, 178th, and 190th is below a given threshold.
    'rest_1vN': None,

    # MvN restraints
    # description:
    #     Contain several 1vN restraints, and only a specific number of them are satisfied.
    #     If there is no such restraints, set to None.
    #     If you have multiple MvN restraints, list them in [].
    #     The order number starts from 1.
    # example:
    #     'rest_MvN': [[10, list(range(160, 170))],
    #                  [78, list(range(160, 170))],
    #                  [120, list(range(160, 170))],
    #                  2]
    #     2 of the 3 given 1vN restraints should be satisfied.
    'rest_MvN': [[[56, list(range(174, 181)) + list(range(197, 220))],
                 [57, list(range(174, 181)) + list(range(197, 220))],
                 [58, list(range(174, 181)) + list(range(197, 220))],
                 [59, list(range(174, 181)) + list(range(197, 220))],
                 [60, list(range(174, 181)) + list(range(197, 220))],
                 [61, list(range(174, 181)) + list(range(197, 220))],
                 [62, list(range(174, 181)) + list(range(197, 220))],
                 [63, list(range(174, 181)) + list(range(197, 220))],
                 [67, list(range(174, 181)) + list(range(197, 220))],
                 [86, list(range(174, 181)) + list(range(197, 220))],
                 [89, list(range(174, 181)) + list(range(197, 220))],
                 [90, list(range(174, 181)) + list(range(197, 220))],
                 [92, list(range(174, 181)) + list(range(197, 220))],
                 [93, list(range(174, 181)) + list(range(197, 220))],
                 [96, list(range(174, 181)) + list(range(197, 220))],
                 [97, list(range(174, 181)) + list(range(197, 220))],
                 [100, list(range(174, 181)) + list(range(197, 220))],
                 10],
                [[174, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [175, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [176, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [177, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [178, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [179, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [180, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [197, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [198, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [199, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [200, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [201, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [202, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [203, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [204, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [205, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [206, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [207, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [208, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [209, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [210, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [211, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [212, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [213, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [214, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [215, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [216, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [217, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [218, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 [219, list([56, 57, 58, 59, 60, 61, 62, 63, 67, 86, 89, 90, 92, 93, 96, 97, 100])],
                 10]],

    # the threshold of the repulsive restraints
    # Change to other values if you know the threshold of the restraints you provide.
    # Due to the definition of distogram in AF2, threshold should be set to a value between 2Å and 22Å
    'rep_thres': 2.0,

    # repulsive restraints
    # description:
    #     The distance between two residues is above a given threshold (rep_thres).
    #     If there is no such restraints, set to None.
    #     If you have multiple repulsive restraints, list them in [].
    #     The order number starts from 1.
    # example:
    #     'rest_rep: [154, 250]
    #     The distance between 154th and 250th residue is above a given threshold
    'rest_rep': None,

    ###########################################################################################################
    # optimization parameters
    ###########################################################################################################
    # if in segment based optimization, set to the length of the segment, for example 200.
    # segment based optimization can save GPU memory, but may lead to suboptimal performance.
    # if not, set to None
    'crop_len': None,

    # the number of rounds to perform
    # large rounds can achive better performance but lead to longer time.
    'rounds': 10,

    # the number of backpropogations in each round
    # if in segment based optimization, set to larger value, for example 150.
    # if not, usually it will converge within 50 steps
    'steps': 50,

    # Save one conformtion in every save_every_n_step step.
    # useful in segment based optimization, since the number of steps is larger
    # and saving conformations in every step will take too much time.
    # if in segment based optimization, set to larger value, for example 3.
    # if not, set to 1.
    'save_every_n_step': 1,

    ###########################################################################################################
    # AF2 model
    ###########################################################################################################
    # AF2 weights dir
    'data_dir': '../params',

    # whether use AF2 in bfloat
    'bfloat': True,
}

config = ml_collections.ConfigDict(config)
