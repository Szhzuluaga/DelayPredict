import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import joblib
warnings.filterwarnings('ignore')
from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression

class DelayModel:

    def __init__(self):
        self._model = LogisticRegression()  # Model should be saved in this attribute.


    def preprocess(self, data: pd.DataFrame, target_column: str = None) \
            -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """



        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')],
            axis=1
        )
        top_10_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        features = features[top_10_features]
        if target_column is not None:


            target = pd.DataFrame(data[target_column])
            return features, target
        else:
            return features


    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.

        """
        target = target.squeeze()
        x_train2, x_test2, y_train2, y_test2 = train_test_split(features, target, test_size = 0.33, random_state = 42)

        n_y0 = len(y_train2[y_train2 == 0])
        n_y1 = len(y_train2[y_train2 == 1])

        self._model = LogisticRegression(max_iter=2000,class_weight={1: n_y0 / len(y_train2), 0: n_y1 / len(y_train2)})
        self._model.fit(x_train2, y_train2)

        self._model_preds_2 = self._model.predict(x_test2)
        print(self._model_preds_2)
        confusion_matrix(y_test2, self._model_preds_2)

        print(classification_report(y_test2, self._model_preds_2))
        joblib.dump(self, 'Modelo_entrenado.pkl')


        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """

        predictions = self._model.predict(features)
        return predictions.tolist()

    def preprocess_for_serving(self,data:pd.DataFrame)->pd.DataFrame:
        top_10_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        data2 = pd.DataFrame(0, columns=top_10_features, index=range(len(data["OPERA"])))
        for col in top_10_features:
            if col in datatest:
                data2[col] = data[col]
        return data2


def get_period_day(date):
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    morning_min = datetime.strptime("05:00", '%H:%M').time()
    morning_max = datetime.strptime("11:59", '%H:%M').time()
    afternoon_min = datetime.strptime("12:00", '%H:%M').time()
    afternoon_max = datetime.strptime("18:59", '%H:%M').time()
    evening_min = datetime.strptime("19:00", '%H:%M').time()
    evening_max = datetime.strptime("23:59", '%H:%M').time()
    night_min = datetime.strptime("00:00", '%H:%M').time()
    night_max = datetime.strptime("4:59", '%H:%M').time()

    if (date_time > morning_min and date_time < morning_max):
        return 'mañana'
    elif (date_time > afternoon_min and date_time < afternoon_max):
        return 'tarde'
    elif (
            (date_time > evening_min and date_time < evening_max) or
            (date_time > night_min and date_time < night_max)
    ):
        return 'noche'


def is_high_season(fecha):
    fecha_año = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
    range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
    range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
    range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
    range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
    range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
    range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
    range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)

    if ((fecha >= range1_min and fecha <= range1_max) or
            (fecha >= range2_min and fecha <= range2_max) or
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
        return 1
    else:
        return 0

def get_min_diff(data):
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds())/60
    return min_diff


#../data/data.csv
data = pd.read_csv('data_with_delay.csv', low_memory=False)

Delay_model = DelayModel()


data['period_day'] = data['Fecha-I'].apply(get_period_day)
data['min_diff'] = data.apply(get_min_diff, axis = 1)

threshold_in_minutes = 15
data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

data.to_csv('data_with_delay.csv', index=False)
features, target = Delay_model.preprocess(data, target_column="delay")

Delay_model.fit(features,target)

datatest = {
    "OPERA": ["American Airlines"],
    "TIPOVUELO": ["I"],
    "MES": [1]
}
dt = pd.DataFrame(datatest)
dt = Delay_model.preprocess_for_serving(dt)

predictions = Delay_model.predict(dt)

print(predictions)
