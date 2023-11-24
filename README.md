# Описание проекта
В данном проекте были рассмотрены различные модели регрессии для предсказания стоимости автомобилей, а также реализован веб-сервис для применения построенной модели на новых данных.

# Краткий справочник по файлам:
HW1_Regression.ipynb    - весь код с EDA и т.п.
main.py                 - реализация микросервиса
encoder.pkl             - сохраненные веса для кодирования признаков
scaler.pkl              - сохраненные веса для стандартизации
grid_search_model.pkl   - сохраненные веса модели
test_data2.csv          - пример файл для тестирования сервиса
метод predict item.PNG  - результаты тестирования по одной записи
метод predict itemS.PNG - результаты тестирования по файлу с записями

# Что было сделано для итоговой модели проекта:
* была использована модель гребневой регрессии
* подобран оптимальный коэффициент регуляризации
* произведено кодирование категориальных признаков
* стандартизация числовых данных
* добавление новых фич (квадраты числовых признаков) - дали наибольший буст

# Результаты:
R^2 = 0.71