# CNY/RUB Rate Prediction ML Pipeline

Полный ML-пайплайн для предсказания курса валютной пары **юань/рубль (CNY/RUB)**:
сбор данных с Московской биржи → feature engineering → обучение модели (LightGBM)
→ сервинг через FastAPI → контейнеризация → деплой в Kubernetes через Helm →
GitOps через ArgoCD → CI/CD через GitHub Actions.

## Источник данных: MOEX, а не Yahoo Finance

С 2022 года CNY/RUB торгуется напрямую на **Московской бирже** (инструмент
`CNYRUB_TOM`, режим `CETS`, расчёты "завтра") — это первичный, ликвидный рынок
для данной пары. Пайплайн использует **MOEX ISS API**
(`https://iss.moex.com/iss/reference/`) как основной источник дневных свечей —
регистрация и токен не нужны.

Yahoo Finance тоже отдаёт кросс-курс `CNYRUB=X`, но это синтетическая величина,
посчитанная через доллар как мост (CNY→USD→RUB), а не реальная котировка, по
которой исполняются сделки. Поэтому Yahoo оставлен в пайплайне только как
резервный источник (`--source yahoo`) — на случай недоступности MOEX ISS или
для экспериментов на других инструментах.

## Почему LightGBM

Курс валютной пары предсказывается на основе табличных признаков (лаги
доходности, скользящие средние, RSI, MACD, волатильность, оборот в рублях).
Для такого типа данных:

- **LightGBM** обучается за секунды-минуты, не требует GPU, устойчив к шуму и
  пропускам, даёт feature importance "из коробки".
- LSTM / трансформеры теоретически лучше ловят long-range зависимости, но на
  дневных данных с ограниченным числом наблюдений почти всегда переобучаются
  и не дают прироста качества относительно бустинга, при этом сильно
  усложняют инфраструктуру (GPU, батчи, нормализация состояний).
- Модель предсказывает **не сам курс**, а **логарифмический return на
  следующий торговый день** — это стационарный таргет, с которым бустинг
  работает существенно лучше, чем с "уезжающим" курсом напрямую.

Важная оговорка: валютные пары, в среднем, куда ближе к случайному блужданию,
чем акции отдельных компаний — макро-новости, действия ЦБ и геополитика часто
значат для курса больше, чем технические индикаторы. Не стоит ждать высокой
directional accuracy (обычно это 51-55% на дневном горизонте) и уж тем более
воспринимать прогноз модели как торговую рекомендацию.

## Структура репозитория

```
.
├── src/
│   ├── config.py              # конфигурация (тикер MOEX, борд, гиперпараметры)
│   ├── data_ingestion.py      # загрузка свечей с MOEX ISS (+ резервный Yahoo)
│   ├── feature_engineering.py # технические индикаторы, лаги, таргет
│   ├── train.py                # обучение + сохранение модели и метрик
│   ├── evaluate.py             # quality gate для CI
│   └── serve.py                 # FastAPI inference-сервис
├── tests/
│   └── test_features.py
├── requirements/
│   ├── train.txt
│   └── serve.txt
├── Dockerfile.train             # образ для обучения (можно запускать как Job в k8s)
├── Dockerfile.serve             # образ для инференса (Deployment в k8s)
├── helm/cnyrub-predictor/        # Helm-чарт для деплоя serving-сервиса
├── argocd/application.yaml       # ArgoCD Application (GitOps)
└── .github/workflows/ci-cd.yaml  # CI: тесты, обучение, сборка образов, обновление Helm values
```

## Как это работает end-to-end

1. **GitHub Actions** при пуше в `main` (и по расписанию каждый будний день):
   - прогоняет тесты (`pytest`);
   - обучает модель на свежих данных с MOEX (`train.py`), логирует метрики;
   - если качество (RMSE на hold-out) не хуже порога — собирает Docker-образ
     `serve` с моделью внутри, пушит в GHCR;
   - обновляет тег образа в `helm/cnyrub-predictor/values.yaml` и коммитит в репозиторий.
2. **ArgoCD** отслеживает `helm/cnyrub-predictor` в Git-репозитории и автоматически
   синхронизирует изменения в кластер Kubernetes (Git — единственный источник
   истины, ArgoCD сам "подтягивает" изменения, откатывает ручные правки мимо Git).
3. **Kubernetes**: Deployment с FastAPI-сервисом инференса, HPA по CPU,
   Service + Ingress, ConfigMap с тикером/источником данных/гиперпараметрами.

## Быстрый старт локально

```bash
pip install -r requirements/train.txt
python src/train.py --ticker CNYRUB_TOM --start 2018-01-01 --source moex
python src/evaluate.py --metrics-path artifacts/metrics.json

pip install -r requirements/serve.txt
uvicorn src.serve:app --reload --app-dir src
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"ticker": "CNYRUB_TOM"}'
```

Пример ответа `/predict`:

```json
{
  "ticker": "CNYRUB_TOM",
  "predicted_log_return": 0.0031,
  "predicted_direction": "up",
  "last_rate": 12.45,
  "predicted_rate_estimate": 12.4887
}
```

## Деплой в кластер

```bash
# 1. Собрать и запушить образ (обычно делает CI, но можно руками)
docker build -f Dockerfile.serve -t ghcr.io/<org>/cnyrub-predictor:latest .
docker push ghcr.io/<org>/cnyrub-predictor:latest

# 2. Зарегистрировать ArgoCD Application
kubectl apply -f argocd/application.yaml

# 3. Дальше ArgoCD сам синхронизирует helm/cnyrub-predictor из Git
```

## Переобучение

Переобучение запускается либо по расписанию (см. `.github/workflows/ci-cd.yaml`,
cron-триггер по будням, т.к. валютный рынок MOEX не торгует по выходным), либо
вручную (`workflow_dispatch`). Новая модель упаковывается в новый Docker-образ
(тег = git SHA), тег прописывается в `values.yaml`, ArgoCD видит diff и
раскатывает новую версию — без ручных `kubectl apply`.

## Смена инструмента / источника данных

Тикер и источник вынесены в конфиг (`src/config.py`, переменные окружения
`TICKER`, `DATA_SOURCE`, `MOEX_BOARD`) и в Helm `values.yaml` — чтобы
переключиться на другую валютную пару, торгуемую на MOEX (например,
`USD000UTSTOM` для доллара или `EUR_RUB__TOM` для евро), достаточно поменять
`TICKER` без изменений в коде.
