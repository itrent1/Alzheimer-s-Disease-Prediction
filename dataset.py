import opendatasets as od
import pandas as pd
import os

dataset_url = 'https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers'

def download_data():
    print("Проверка наличия данных...")
    od.download(dataset_url)
    
    data_dir = 'alzheimers-disease-dataset'
    file_name = 'alzheimers_disease_data.csv'
    path = os.path.join(data_dir, file_name)
    
    if os.path.exists(path):
        print(f"Данные успешно загружены: {path}")
        return path
    else:
        print("Ошибка: файл не найден.")
        return None

if __name__ == "__main__":
    file_path = download_data()
    if file_path:
        df = pd.read_csv(file_path)
        print(df.head())