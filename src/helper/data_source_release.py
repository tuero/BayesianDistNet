"""
File: data_source_release.py
Date: May 10, 2020
Description: Data source handling
Source: https://github.com/KEggensperger/DistNet/
"""

import os


def get_data_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")


def get_sc_dict():
    data_dir = get_data_dir()
    sc_dict = {
        "clasp_factoring":
        {"scen": "clasp-3.0.4-p8_rand_factoring",
            "features": "{}/clasp-3.0.4-p8_rand_factoring/features.txt".format(data_dir),
            "domain": "sat",
            "use": ('SAT',),
            "cutoff": 5000},
        "saps-CVVAR":
        {"scen": "CP06_CV-VAR",
            "features": "{}/CP06_CV-VAR/features.txt".format(data_dir),
            "domain": "sat",
            "use": ('SAT',),
            "cutoff": 60},
        "spear_qcp":
        {"scen": "spear_qcp-hard",
            "features": "{}/spear_qcp-hard/features.txt".format(data_dir),
            "domain": "sat",
            "use": ('SAT',),
            "cutoff": 30},
        "yalsat_qcp":
        {"scen": "yalsat_qcp-hard",
            "features": "{}/yalsat_qcp-hard/features.txt".format(data_dir),
            "domain": "sat",
            "use": ('SAT',),
            "cutoff": 30},
        "spear_swgcp":
        {"scen": "spear_smallworlds",
            "features": "{}/spear_smallworlds/features.txt".format(data_dir),
            "domain": "sat",
            "use": ('SAT',),
            "cutoff": 15},
        "yalsat_swgcp":
        {"scen": "yalsat_smallworlds",
            "features": "{}/yalsat_smallworlds/features.txt".format(data_dir),
            "domain": "sat",
            "use": ('SAT',),
            "cutoff": 5000},
        "lpg-zeno":
        {"scen": "lpg-zenotravel",
            "features": "{}/lpg-zenotravel/features.txt".format(data_dir),
            "domain": "planning",
            "use": ('SAT', ),
            "cutoff": 300}
    }

    return sc_dict
