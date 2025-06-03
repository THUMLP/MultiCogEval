import re
from typing import Dict

REGION_EXACT_DICT = {"Abdomen": ["gi", "eus", "mrcp", "hida", "ercp"],'Chest':['cxt']}

REGION_SUBSTR_DICT = {
    "Chest": [
        "chest",
        "lung",
        "upper lobe",
        "lower lobe",
        "pleura",
        "atelectasis",
        "ground.glass",
        "heart",
        "cardiac",
        "pericard",
        "mediastin",
        "pneumothorax",
        "breast",
        "pulmonary",
        "coronary arteries",
        "coronary angiogram",
        "coronary artery",
        "thorax",
        'myocardial'
    ],
    "Abdomen": [
        "abd",
        "abdom",
        "pelvi",
        "liver",
        "gallbladder",
        "pancrea",
        "duct",
        "spleen",
        "stomach",
        "bowel",
        "rectum",
        "ileum",
        "iliac",
        "duodenum",
        "colon",
        "urinary",
        "bladder",
        "ureter",
        "kidney",
        "renal",
        "adrenal glands",
        "intraperitoneal",
        "ascites",
        "prostate",
        "uterus",
        "appendi",
        "retroperit",
        "mesenter",
        "paracolic",
        "lower quadrant",
        "perirectal",
        "cul-de-sac",
        "iliopsoas",
        "psoas",
        "hepatic",
        "hepato",
        "quadrant",
        "gastro",
        "biliary"
    ],
    "Venous": [
        "venous",
        "jugular",
        "cephalic",
        "axillary",
        "basilic",
        "brachial",
        "popliteal",
        "femoral",
        "peroneal",
        "tibial",
        "fibular",
        "veins",
    ],
    "Head": ["head", "brain", "skull","orbit","mastoid",'face','cerebral','cerebellum'],
    "Neck": ["neck", "thyroid"],
    "Scrotum": ["scrot", "testic"],
    "Spine": ["spine", "cervical",'thoracic','spinal'],
    "Ankle": ["ankle",'lower extremities'],
    "Foot": ["foot",'lower extremities'],
    "Bone": ["bone"],
    "Knee": ["knee",'lower extremities'],
    "Hand": ["hand"],
    "Wrist": ["wrist"],
    "Finger": ["finger"],
    "Heel": ["heel",'lower extremities'],
    "Hip": ["hip",'lower extremities'],
    "Shoulder": ["shoulder"],
    "Thigh": ["thigh",'lower extremities'],
    "Femur": ["femur"],
    #'Extremity' : ['extremity', 'arm', 'leg', 'thigh', 'knee', 'hands']
    # "Upper Extremity": ["up(.*)ext"],
    # "Lower Extremity": ["low(.*)ext"], lower extremities
}

MODALITY_EXACT_DICT = {
    "CT": ["ct", "cat", "cta", "mdct", "ctu", "dlp", "mgy",'computed tomography'],
    "Ultrasound": ["us", "dup",'usg'],
    "Radiograph": ["ap", "pa", "cxr"],
    "MRI": ["mri", "mr", "mrcp", "t\d"],
}

MODALITY_SUBSTR_DICT = {
    "CT": ["multidetector", "reformat", "optiray"],
    "Ultrasound": [
        "u\.s\.",
        "ultrasound",
        "echotexture",
        "sonogra",
        "doppler",
        "duplex",
        "doppler",
        "echogenic",
        "transabdominal",
        "transvaginal",
        "non-obstetric",
    ],
    "Radiograph": [
        "port\.",
        "radiograph",
        "portable",
        "x-ray",
        "supine and",
        "supine &",
        "supine only",
        "and lateral",
        "frontal view",
        "supine view",
        "single view",
        "two views",
        "angiography"
    ],
    "MRI": ["gadavist", "magnet", "tesla"],
    "Fluoroscopy": ["fluoro"],
}

MODALITY_SPECIAL_CASES_DICT = {
    'cardiac catheterization': ['cardiac.*catheterization'],
    "CAD": ["coronary.*angiogram", "coronary.*artery.*disease",'cad','coronary.*angiography'],
    "DSM": ["dsm", "digital.*subtraction.*myelogram"],
    "ECG": [
        'ELECTROCARDIOGRAM',
        'ecg'
    ],
    "DTI": [
        'DIFFUSION TENSOR IMAGING',
        'dti'
    ],
    "ECHO": [
        'ECHOCARDIOGRAM',
        'echo'
    ],
    "CTU": [
        "ctu",
        "ct urogram",
        "ct urography",
        "ct ivu",
        "ct ivp",
        "ct intravenous pyelography",
    ],
    'CTPA': ['ctpa','computed.*tomography.*pulmonary.*angiogram'],
    "Drainage": ["drain"],
    "Carotid ultrasound": ["carotid.*ultrasound", "carotid.*us", "carotid.*series"],
    "EUS": ["eus", "endoscopic.*(ultrasound|us)", "echo.*endoscopy"],
    "MRCP": ["mrcp", "magnetic.*resonance.*cholangiopancreatography"],
    "HIDA": ["hida", "hepatobiliary.*iminodiacetic.*acid"],
    "ERCP": [
        "ercp",
        "endoscopic.*retrograde.*cholangiopancreatography",
        "bil endoscopy",
    ],
    "PTC": [
        "ptc",
        "percutaneous.*transhepatic.*cholangiography",
        "perc transhepatic cholangiography",
    ],
    "Upper GI Series": [
        "upper.*gi",
        "upper.*gastrointestinal",
        "barium.*swallow",
        "barium.*meal",
        "barium.*study",
        "ugis",
        "bas/ugi",
    ],
    "Lower GI Series": [
        "lower.*gi",
        "lower.*gastrointestinal",
        "barium.*swallow",
        "barium.*meal",
        "barium.*study",
        "lgis",
        "bas/lgi",
    ],
    "Paracentesis": ["paracentesis"],
    "Mammogram": ["mammo"],
    "MRA": ["mra", "magnetic.*resonance.*angiography"],
    "MRE": ["mre", "magnetic.*resonance.*enterography", "mr enterography"],
    'VCUG': ['vcug','voiding.*cystourethrogram'],
}

UNIQUE_TO_BROAD_MODALITY = {
    "DSM": "Radiograph",
    "CTU": "CT",
    "Carotid ultrasound": "Ultrasound",
    "EUS": "Ultrasound",
    "MRCP": "MRI",
    "ERCP": "Radiograph",
    "Upper GI Series": "Radiograph",
    "Lower GI Series": "Radiograph",
    "MRA": "MRI",
    "MRE": "MRI",
    "ECG": "Radiograph",
    "ECHO": "Ultrasound",
    'DTI': 'MRI',
    'CAD': 'Radiograph',
    'CTPA': 'CT',
    'VCUG': 'Radiograph',
    'cardiac catheterization': 'Radiograph'
}

UNIQUE_MODALITY_TO_ORGAN_MAPPING = {
    "CTU": "Abdomen",
    "EUS": "Abdomen",
    "MRCP": "Abdomen",
    "HIDA": "Abdomen",
    "ERCP": "Abdomen",
    "MRE": "Abdomen",
    "Upper GI Series": "Abdomen",
    "Lower GI Series": "Abdomen",
    "Carotid ultrasound": "Neck",
    "Mammogram": "Chest",
    "ECG": "Chest",
    "ECHO": "Chest",
    'DTI': 'Head',
    "DSM": "Spine",
    'CAD': 'Chest',
    'CTPA': 'Chest',
    'VCUG': 'Abdomen',
    'cardiac catheterization': 'Chest'
}

def identify_entity_cat(
    text, exact_dict: Dict = {}, substr_dict: Dict = {}, special_cases_dict: Dict = {}
):
    # counts = {}
    return_set = set()
    # If there is a special cases match, return that
    for category, patterns in special_cases_dict.items():
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if len(matches) > 0:
                return_set.add(category)
                return list(return_set)

    # Initialize counts for all categories
    # for cat in set(list(exact_dict.keys()) + list(substr_dict.keys())):
    #     counts[cat] = 0

    # Count exact matches
    for category, words in exact_dict.items():
        for word in words:
            pattern = r"\b" + word + r"\b"
            matches = re.findall(pattern, text, re.IGNORECASE)
            if len(matches) > 0:
                return_set.add(category)

    # Count substring matches
    for category, substrings in substr_dict.items():
        for pattern in substrings:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if len(matches) > 0:
                return_set.add(category)

    return list(return_set)

def get_modality(text):
    return identify_entity_cat(
        text,
        exact_dict=MODALITY_EXACT_DICT,
        substr_dict=MODALITY_SUBSTR_DICT,
        special_cases_dict=MODALITY_SPECIAL_CASES_DICT,
    )
def get_region(text, modality):
    regions = identify_entity_cat(
        text,
        exact_dict=REGION_EXACT_DICT,
        substr_dict=REGION_SUBSTR_DICT,
    )
    if len(regions) == 0:
        for mod in modality:
            if mod in UNIQUE_MODALITY_TO_ORGAN_MAPPING:
                regions = [UNIQUE_MODALITY_TO_ORGAN_MAPPING[mod]]
                break
    return regions

def parse_radiology_request(text):
    modality = get_modality(text)
    region = get_region(text, modality)
    return modality, region
        # if modality in UNIQUE_MODALITY_TO_ORGAN_MAPPING:
        #     frequent_region = UNIQUE_MODALITY_TO_ORGAN_MAPPING[frequent_modality]
        #     frequent_region_count = 1