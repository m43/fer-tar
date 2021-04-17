import ds_load

DS_PATH = './dataset/essays.csv'

if __name__ == '__main__':
    dataset = ds_load.load_dataset(DS_PATH)
    print("Personality Trait Classification on Essays")
    print("Text Analysis and Retrieval, 2021")
    print("FER, University of Zagreb, Croatia")
