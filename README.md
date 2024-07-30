# Flood_Hackathon

## Использование

1. Убедитесь, что у вас есть следующие файлы в директории проекта:
- `water_level_model.joblib`
- `scaler.joblib`
- `feature_selector.joblib`

2. Создайте Python-скрипт `predict.py` со следующим содержимым:

```python
import joblib
import numpy as np
import pandas as pd

# Загрузка модели и вспомогательных объектов
model = joblib.load('water_level_model.joblib')
scaler = joblib.load('scaler.joblib')
rfe = joblib.load('feature_selector.joblib')

def predict_next_days(model, scaler, rfe, last_data, days=7):
 predictions = []
 current_data = last_data.copy()
 
 # Получаем названия признаков, использованных при обучении
 train_features = rfe.get_support()
 
 # Убеждаемся, что все необходимые признаки присутствуют
 for feature in train_features:
     if feature not in current_data.columns:
         # Если признак отсутствует, добавляем его с значением по умолчанию
         current_data[feature] = 0
 
 for _ in range(days):
     # Используем только признаки, которые были при обучении
     current_data_for_prediction = current_data[train_features]
     
     current_data_scaled = scaler.transform(current_data_for_prediction)
     prediction = model.predict(current_data_scaled)
     predictions.append(prediction[0])
     
     # Обновляем данные для следующего прогноза
     current_data['water_level'] = prediction[0]
     
     # Обновляем другие признаки по необходимости
     if 'water_level_rolling_mean' in train_features:
         current_data['water_level_rolling_mean'] = (current_data['water_level_rolling_mean'] * 6 + prediction[0]) / 7
     if 'water_level_rolling_std' in train_features:
         current_data['water_level_rolling_std'] = np.sqrt(((current_data['water_level_rolling_std']**2 * 6) + 
                                                            (prediction[0] - current_data['water_level_rolling_mean'])**2) / 7)
     if 'day_of_week' in train_features:
         current_data['day_of_week'] = (current_data['day_of_week'] + 1) % 7
     
     if 'water_level_flood' in train_features:
         current_data['water_level_flood'] = 0
 
 return predictions

# Пример использования
last_known_data = pd.DataFrame({
 'water_level': [100],
 'water_level_rolling_mean': [95],
 'water_level_rolling_std': [5],
 'day_of_week': [0],
 'water_level_flood': [0],
 # Добавьте другие необходимые признаки
})

future_predictions = predict_next_days(model, scaler, rfe, last_known_data)
print("Прогноз на следующие 7 дней:", future_predictions)
