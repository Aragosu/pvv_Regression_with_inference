# venv\scripts\activate
# pip install fastapi[all]
# python.exe -m pip install --upgrade pip
# pip freeze > requirements.txt
# uvicorn main:app --reload


from fastapi import FastAPI,HTTPException, UploadFile, File, Response
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import io
from fastapi.responses import StreamingResponse

import warnings
warnings.filterwarnings('ignore')

from joblib import dump, load

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
#    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


# ==============================  0. Загрузка данных  ==============================
def preprocessing_f(start_df, type_df='new'):

    # ==============================  1. Препроцессинг данных - чистка ==============================
    # 1.1 - старт
    new_df_train = start_df

    # 1.2 - Обработка признаков
    # трейн
    new_df_train['mileage'] = new_df_train['mileage'].str.extract(r'(\d+.\d+|\d+)').astype(float)
    new_df_train['engine'] = new_df_train['engine'].str.extract(r'(\d+.\d+|\d+)').astype(float)
    new_df_train['max_power'] = new_df_train['max_power'].str.extract(r'(\d+.\d+|\d+)').astype(float)

    # 1.3 - Удаление "лишних" столбцов
    new_df_train = new_df_train.drop('torque', axis=1)

    # 1.4 - заполнение пустых значений
    mileage_median = round(new_df_train.mileage.median(), 2)
    engine_median = round(new_df_train.engine.median(), 1)
    max_power_median = round(new_df_train.max_power.median(), 2)
    seats_median = round(new_df_train.seats.median(), 1)

    new_df_train['mileage'] = new_df_train['mileage'].fillna(mileage_median)
    new_df_train['engine'] = new_df_train['engine'].fillna(engine_median)
    new_df_train['max_power'] = new_df_train['max_power'].fillna(max_power_median)
    new_df_train['seats'] = new_df_train['seats'].fillna(seats_median)

    # 1.5 - форматы
    # трейн
    new_df_train['engine'] = new_df_train['engine'].astype(int)
    new_df_train['seats'] = new_df_train['seats'].astype(int)
    new_df_train['km_driven'] = new_df_train['km_driven'].astype(int)

    # ==============================  2. Препроцессинг данных - добавление фичей  ==============================
    # 2.1 - квадраты числовых параметров
    new_df_train['year_2'] = new_df_train['year'] ** 2
    new_df_train['km_driven_2'] = new_df_train['km_driven'] ** 2
    new_df_train['mileage_2'] = new_df_train['mileage'] ** 2
    new_df_train['engine_2'] = new_df_train['engine'] ** 2
    new_df_train['max_power_2'] = new_df_train['max_power'] ** 2

    # 2.2 - Выделяем X,Y
    X_train_fin = new_df_train[['year', 'km_driven', 'fuel', 'seller_type',
                                'transmission', 'mileage', 'engine', 'max_power', 'seats',  # 'mark_name',
                                'year_2', 'km_driven_2', 'mileage_2', 'engine_2',
                                'max_power_2']]
    return X_train_fin


# ==============================  3. Кодировка  ==============================
def encoder_f(X_start):
    X_cat = X_start[['fuel','seller_type','transmission']]
    encoder_loaded = load('encoder.pkl')
    encoded_data_array = encoder_loaded.transform(X_cat).toarray()
    encoded_df = pd.DataFrame(encoded_data_array,
                              columns = encoder_loaded.get_feature_names_out(list(X_cat.columns))).reset_index(drop=True)
    X_fin = X_start.drop(X_cat, axis=1).reset_index(drop=True)
    X_encoded = pd.concat([X_fin, encoded_df], axis=1)
    return X_encoded

# ==============================  4. Стандартизация  ==============================
def scaler_f(X):
    scaler_loaded = load('scaler.pkl')
    X_encoded_scaled = scaler_loaded.transform(X)
    X_encoded_scaled = pd.DataFrame(X_encoded_scaled,columns = X.columns)
    return X_encoded_scaled

# ==============================  5. FastAPI сервис  ==============================

@app.get('/')
def root():
    return "Привет пользователь"

@app.post("/predict_item")
async def predict_item(item: Item) -> float:
    df_new = pd.DataFrame([item.dict()])
    X_new = preprocessing_f(df_new, type_df='test')
    X_new_encoded = encoder_f(X_new)
    X_new_encoded_scaled = scaler_f(X_new_encoded)
    model_loaded = load('grid_search_model.pkl')
    Y_new_pred = model_loaded.predict(X_new_encoded_scaled)
    return Y_new_pred[0][0]

@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    # Считывание файла CSV и создание датафрейма
    df_new = pd.read_csv(file.file,dtype={'name': str,
                                      'year': int,
                                      'km_driven': int,
                                      'fuel': str,
                                      'seller_type': str,
                                      'transmission': str,
                                      'owner': str,
                                      'mileage': str,
                                      'engine': str,
                                      'max_power': str,
                                      'torque': str,
                                      'seats': float})
    X_new = preprocessing_f(df_new, type_df='test')
    X_new_encoded = encoder_f(X_new)
    X_new_encoded_scaled = scaler_f(X_new_encoded)
    model_loaded = load('grid_search_model.pkl')
    Y_new_pred = model_loaded.predict(X_new_encoded_scaled)
    Y_new_pred = pd.DataFrame(Y_new_pred, columns=['pred'])
    result_df = pd.concat([df_new, Y_new_pred], axis=1)

    stream = io.StringIO()
    result_df.to_csv(stream, index=False)
    response = StreamingResponse(
        iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"

    return response