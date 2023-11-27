Альфа версия веб-приложения "ОНЛАЙН Рентгенолог" является системой помощи принятия врачебных решений (СППВР), построенное на основе модели классификации - сверточной нейронной сети, обученной методом глубокого машинного обучения с учителем.

**Цель приложения** - второе чтение рентгенограмм ОГК и флюорограмм с целью помощи врачу-рентгенологу в определении наличия или отсутствия признаков вирусной пневмонии (COVID-19) в легких.

**Результатом работы** нейронной сети является предсказание по снимку ОГК наиболее вероятного класса (normal или opacity), а также вероятность данного класса.

Веб-приложение написано на языке Python v3.11.6 с помощью библиотеки Streamlit v1.28.2. Сверточная нейронная сеть разработана с использованием библиотеки PyTorch v2.1.1+cpu в облачной среде разработки Google Colab.

Для запуска приложения необходимо перейти по адресу `url`, в область загрузки загрузить изображение в формате jpg. После того, как загруженно изображение отобразится на экране, нажать кнопку "Обработать". Внизу снимка появится предсказание в виде: "**Предсказанный класс:** opacity , **вероятность: 0.84**"
