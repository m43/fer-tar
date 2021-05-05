import re

DS_PATH = './dataset/essays.csv'
TRAITS = ['ext', 'neu', 'agr', 'con', 'opn']

R_ID = r'\d{4}_\d+\.txt'
R_TEXT = r'"?.+"?'
R_BIG5 = r'[n|y]'
REGEX_ROW = f"[^#]?({R_ID}),({R_TEXT}),({R_BIG5}),({R_BIG5}),({R_BIG5}),({R_BIG5}),({R_BIG5})"
PATTERN_ROW = re.compile(REGEX_ROW)


def load_dataset():
    """
    Function for loading dataset from .csv file. It reads the .csv file and parses it, thus creating a list of
    all file entries. A entry consists of 7 attributes which are: author, text and bool flags for extroversion,
    neuroticism, agreeableness, conscientiousness, openness.

    :return: list containing list of dataset entries: list[list[string, string, bool, bool, bool, bool, bool]]
    """
    dataset = []
    # loading does not work with 'utf8' for some reason
    with open(DS_PATH, 'r', encoding='cp1252') as essays:
        while True:
            # for large files, line by line is better than reading all lines
            line = essays.readline()
            if not line:
                break

            match = PATTERN_ROW.match(line)
            if match is None:
                continue

            groups = match.groups()
            author, text = groups[0], groups[1]
            c_ext, c_neu, c_agr, c_con, c_opn = [g == 'y' for g in groups[2:]]
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            dataset.append([author, text, c_ext, c_neu, c_agr, c_con, c_opn])
    return dataset


if __name__ == '__main__':
    dataset = load_dataset()
    assert len(dataset) == 2467
    assert len(dataset[0]) == 7
    assert dataset[0][2:] == [False, True, True, False, True]
