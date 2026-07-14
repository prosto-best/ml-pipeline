# Stock Price Prediction ML Pipeline

Полный ML-пайплайн для предсказания цены акции: сбор данных → feature engineering →
обучение модели (LightGBM) → сервинг через FastAPI → контейнеризация → деплой в
Kubernetes через Helm → GitOps через ArgoCD → CI/CD через GitHub Actions.

## Почему LightGBM

Цена акции предсказывается на основе табличных признаков (лаги цены, скользящие
средние, RSI, MACD, волатильность, объём и т.д.). Для такого типа данных:

- **LightGBM** обучается за секунды-минуты, не требует GPU, устойчив к шуму и
  пропускам, даёт feature importance "из коробки".
- LSTM / трансформеры теоретически лучше ловят long-range зависимости, но на
  дневных OHLCV-данных с ограниченным числом наблюдений почти всегда
  переобучаются и не дают прироста качества относительно бустинга, при этом
  сильно усложняют инфраструктуру (GPU, батчи, нормализация состояний).
- Модель предсказывает **не саму цену**, а **логарифмический return на
  следующий торговый день** — это стационарный таргет, с которым бустинг
  работает существенно лучше, чем с трендовой ценой напрямую.

## Структура репозитория

```
stock-price-ml-pipeline/
├── src/
│   ├── config.py              # конфигурация (тикер, даты, гиперпараметры)
│   ├── data_ingestion.py      # загрузка OHLCV через yfinance
│   ├── feature_engineering.py # технические индикаторы, лаги, таргет
│   ├── train.py                # обучение + логирование в MLflow + сохранение модели
│   ├── evaluate.py             # метрики качества на hold-out
│   └── serve.py                 # FastAPI inference-сервис
├── tests/
│   └── test_features.py
├── requirements/
│   ├── train.txt
│   └── serve.txt
├── Dockerfile.train             # образ для обучения (Job в k8s)
├── Dockerfile.serve             # образ для инференса (Deployment в k8s)
├── helm/stock-predictor/        # Helm-чарт для деплоя serving-сервиса
├── argocd/application.yaml       # ArgoCD Application (GitOps)
└── .github/workflows/ci-cd.yaml  # CI: тесты, обучение, сборка образов, обновление Helm values
```

## Как это работает end-to-end

1. **GitHub Actions** при пуше в `main`:
   - прогоняет тесты (`pytest`);
   - обучает модель на свежих данных (`train.py`), логирует метрики;
   - если качество (MAE/RMSE на hold-out) не хуже порога — собирает Docker-образ
     `serve` с моделью внутри, пушит в GHCR;
   - обновляет тег образа в `helm/stock-predictor/values.yaml` и коммитит в репозиторий.
2. **ArgoCD** отслеживает `helm/stock-predictor` в Git-репозитории и автоматически
   синхронизирует изменения в кластер Kubernetes (классический GitOps-паттерн:
   Git — единственный источник истины, ArgoCD сам "подтягивает" изменения).
3. **Kubernetes**: Deployment с FastAPI-сервисом инференса, HPA по CPU,
   Service + Ingress, ConfigMap с гиперпараметрами/тикером.

## Быстрый старт локально

```bash
pip install -r requirements/train.txt
python src/train.py --ticker AAPL --start 2015-01-01
python src/evaluate.py --model-path artifacts/model.joblib

pip install -r requirements/serve.txt
uvicorn src.serve:app --reload
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

## Деплой в кластер

```bash
# 1. Собрать и запушить образ (обычно делает CI, но можно руками)
docker build -f Dockerfile.serve -t ghcr.io/<org>/stock-predictor:latest .
docker push ghcr.io/<org>/stock-predictor:latest

# 2. Зарегистрировать ArgoCD Application
kubectl apply -f argocd/application.yaml

# 3. Дальше ArgoCD сам синхронизирует helm/stock-predictor из Git
```

## Переобучение

Переобучение запускается либо по расписанию (см. `.github/workflows/ci-cd.yaml`,
cron-триггер), либо вручную (`workflow_dispatch`). Новая модель упаковывается в
новый Docker-образ (тег = git SHA), тег прописывается в `values.yaml`, ArgoCD
видит diff и раскатывает новую версию — без ручных kubectl apply.
