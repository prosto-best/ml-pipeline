# CNY/RUB Rate Prediction ML Pipeline

Полный ML-пайплайн для предсказания курса валютных пар MOEX (юань/рубль и
другие): сбор данных → feature engineering → обучение модели (LightGBM) с
трекингом в MLflow → сервинг через FastAPI → мониторинг живого качества и
data drift → контейнеризация → деплой в Kubernetes через Helm → GitOps через
ArgoCD → CI/CD через GitHub Actions.

## Источник данных: MOEX, а не Yahoo Finance

С 2022 года CNY/RUB торгуется напрямую на **Московской бирже** (инструмент
`CNYRUB_TOM`, режим `CETS`) — это первичный, ликвидный рынок для данной пары.
Пайплайн использует **MOEX ISS API** как основной источник дневных свечей,
регистрация не нужна. Yahoo Finance оставлен как резервный источник
(`--source yahoo`).

## Мультитикерный режим

Сервис обучает и обслуживает НЕСКОЛЬКО валютных пар одновременно (по
умолчанию `CNYRUB_TOM`, `USD000UTSTOM`, `EUR_RUB__TOM` — все на MOEX/CETS),
каждая со своей отдельной моделью. Один запрос к `/predict` отдаёт прогнозы
по всем сразу, одним JSON.

## Почему LightGBM

Курс валютной пары предсказывается на основе табличных признаков (лаги
доходности, скользящие средние, RSI, MACD, волатильность). LightGBM
обучается за секунды-минуты, не требует GPU, устойчив к шуму, даёт feature
importance "из коробки" — в отличие от LSTM/трансформеров, которые на
дневных данных с ограниченным числом наблюдений чаще переобучаются, чем
дают реальный прирост качества, при этом сильно усложняют инфраструктуру.

Модель предсказывает **логарифмический return на следующий торговый день**,
а не сам курс — стационарный таргет, с которым бустинг работает лучше.

Важная оговорка: валютные пары в среднем ближе к случайному блужданию, чем
акции — макро-новости и действия ЦБ значат больше, чем технические
индикаторы. Не ждите directional accuracy выше 51-55% и не используйте
прогноз как торговую рекомендацию.

## MLOps-часть: MLflow + мониторинг качества

- **MLflow** (tracking + Model Registry) — каждый запуск `train.py` логирует
  параметры, метрики и саму модель; при прохождении quality gate
  (`evaluate.py`) версия модели автоматически переводится в стадию
  `Production` в Model Registry. Из коробки поднимается в кластере на
  SQLite + PVC (просто, но для serious prod — замените на Postgres + S3,
  см. `helm/cnyrub-predictor/values.yaml`).
- **`src/monitor.py`** — периодическая (CronJob) проверка УЖЕ задеплоенных
  моделей на живых данных:
  - *живое качество*: т.к. горизонт предсказания — 1 день, через день
    фактический результат уже известен. Скрипт пересчитывает RMSE/MAE/
    directional accuracy на последних N реализованных днях и сравнивает с
    качеством при обучении — это честная проверка "в бою", а не только на
    историческом hold-out;
  - *data drift* (**Evidently**) — сравнивает распределение свежих признаков
    с baseline-сэмплом из обучения; существенный дрифт — ранний сигнал,
    что рыночный режим изменился.
  - Результаты пишутся в отдельный MLflow-эксперимент и пушатся в
    **Prometheus Pushgateway** (job-метрики нельзя просто "заскрейпить",
    т.к. CronJob-под завершается и исчезает).
- **Prometheus + Grafana** — serving-сервис отдаёт `/metrics`
  (`prometheus-fastapi-instrumentator`: латентность, RPS, коды ответов) плюс
  кастомные гейджи последнего предсказания на тикер. `ServiceMonitor` в Helm
  подключает это к Prometheus Operator, если он уже есть в кластере.

## Структура репозитория

```
.
├── src/
│   ├── config.py               # конфигурация (тикеры, MLflow, пороги мониторинга)
│   ├── data_ingestion.py       # загрузка свечей с MOEX ISS (+ резервный Yahoo)
│   ├── feature_engineering.py  # технические индикаторы, лаги, таргет
│   ├── train.py                 # обучение всех тикеров + логирование в MLflow
│   ├── evaluate.py              # quality gate + promote в MLflow Production
│   ├── serve.py                  # FastAPI: /predict по всем тикерам + /metrics
│   └── monitor.py                # живой quality-мониторинг + data drift (Evidently)
├── tests/
│   └── test_features.py
├── requirements/
│   ├── train.txt / serve.txt / monitor.txt
├── Dockerfile.train / Dockerfile.serve / Dockerfile.monitor
├── helm/cnyrub-predictor/
│   └── templates/
│       ├── deployment.yaml, service.yaml, hpa.yaml, ingress.yaml, configmap.yaml
│       ├── mlflow-deployment.yaml, mlflow-service.yaml, mlflow-pvc.yaml, mlflow-ingress.yaml
│       ├── monitor-cronjob.yaml    # периодический запуск monitor.py
│       ├── pushgateway.yaml         # приёмник метрик от CronJob
│       └── servicemonitor.yaml       # интеграция с Prometheus Operator
├── argocd/application.yaml
└── .github/workflows/ci-cd.yaml
```

## Как это работает end-to-end

1. **GitHub Actions**: тесты → обучение всех тикеров (с логированием в
   MLflow, если задан секрет `MLFLOW_TRACKING_URI`) → quality gate
   (+ promote в MLflow Registry) → сборка и пуш ДВУХ образов (serving и
   monitor) в GHCR → безопасное обновление обоих тегов в `values.yaml`
   через `yq` (не sed — чтобы не перепутать теги двух разных образов).
2. **ArgoCD** синхронизирует `helm/cnyrub-predictor` в кластер.
3. **Kubernetes**: Deployment serving-сервиса, MLflow-сервер, CronJob
   мониторинга, опционально Pushgateway и ServiceMonitor.

## Быстрый старт локально

```bash
pip install -r requirements/train.txt
python src/train.py --start 2018-01-01 --source moex   # обучит все тикеры из TICKERS
python src/evaluate.py --artifacts-dir artifacts

pip install -r requirements/serve.txt
uvicorn src.serve:app --reload --app-dir src
curl localhost:8000/predict          # прогнозы по всем тикерам сразу
curl localhost:8000/metrics           # метрики Prometheus
```

Пример ответа `/predict`:

```json
{
  "CNYRUB_TOM": {
    "predicted_log_return": 0.0031,
    "predicted_direction": "up",
    "last_rate": 12.45,
    "predicted_rate_estimate": 12.4887
  },
  "USD000UTSTOM": { "...": "..." },
  "EUR_RUB__TOM": { "...": "..." }
}
```

Локальный запуск мониторинга:

```bash
pip install -r requirements/monitor.txt
python src/monitor.py
```

## Деплой в кластер

```bash
docker build -f Dockerfile.serve -t ghcr.io/<org>/cnyrub-predictor:latest .
docker build -f Dockerfile.monitor -t ghcr.io/<org>/cnyrub-monitor:latest .
docker push ghcr.io/<org>/cnyrub-predictor:latest
docker push ghcr.io/<org>/cnyrub-monitor:latest

kubectl apply -f argocd/application.yaml
# Дальше ArgoCD сам синхронизирует helm/cnyrub-predictor из Git
```

MLflow UI будет доступен внутри кластера по адресу
`http://<release>-cnyrub-predictor-mlflow:5000` (включите `mlflow.ingress`
в `values.yaml`, если нужен доступ снаружи — и обязательно закройте его
basic-auth, у MLflow нет встроенной аутентификации).

## HTTPS / TLS в кластере (Traefik IngressRoute + cert-manager, самоподписанный сертификат)

В кластере не используется стандартный Kubernetes `Ingress`/`IngressClass` —
маршрутизация идёт через `IngressRoute` (CRD Traefik). Это меняет то, как
подключается cert-manager: его механизм ingress-shim (автосоздание
`Certificate` по аннотациям на `Ingress`) работает только для обычного
`Ingress` и **не видит** `IngressRoute` — поэтому `Certificate` заводится явно
(`templates/certificate.yaml`), а `IngressRoute` только ссылается на готовый
`Secret` по имени.

**Шаг 1. Установить cert-manager (один раз на кластер, не часть этого чарта):**

```bash
helm repo add jetstack https://charts.jetstack.io
helm repo update
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace \
  --set installCRDs=true
kubectl get pods -n cert-manager   # все 3 пода (controller, cainjector, webhook) должны быть Running
```

**Шаг 2. Ничего больше руками создавать не нужно.** В `values.yaml` уже включено:

```yaml
traefik:
  ingressRoute:
    enabled: true
    host: cnyrub-predictor.62.181.44.191.sslip.io
    entryPoint: websecure   # HTTPS-entrypoint Traefik (обычно порт 443)
    tls:
      enabled: true
      secretName: cnyrub-predictor-tls
      certManager:
        enabled: true
        clusterIssuerName: selfsigned-cluster-issuer
        installClusterIssuer: true   # чарт сам создаст ClusterIssuer
    httpRedirect:
      enabled: true   # автоматический редирект с http:// на https://
```

При `helm upgrade` / синхронизации ArgoCD чарт создаст:
- `ClusterIssuer` типа `selfSigned` (`templates/cluster-issuer.yaml`);
- `Certificate`, ссылающийся на этот issuer (`templates/certificate.yaml`) —
  cert-manager увидит его и положит выпущенный сертификат в
  `Secret` `cnyrub-predictor-tls`;
- `IngressRoute` на entrypoint `websecure`, который берёт TLS именно из этого
  `Secret` (`templates/ingressroute.yaml`);
- второй `IngressRoute` + `Middleware` на entrypoint `web` (HTTP, 80), который
  ничего не отдаёт сам, а только редиректит на `https://` — без него голый
  HTTP-трафик продолжил бы обслуживаться в обход TLS.

**Проверка:**

```bash
curl -k https://cnyrub-predictor.62.181.44.191.sslip.io/predict
curl -IL http://cnyrub-predictor.62.181.44.191.sslip.io/predict   # должен вернуть 301/308 на https

kubectl get certificate -A                # READY=True, когда сертификат выпущен
kubectl describe certificate <name> -n <ns>   # если долго не READY -- смотреть сюда
kubectl describe certificaterequest -A
```

Флаг `-k` у curl обязателен: самоподписанный сертификат не подписан ни одним
публичным CA, поэтому curl/браузер честно скажут "untrusted certificate" — это
ожидаемо и нормально для такого сценария (канал при этом реально шифруется
TLS, просто без цепочки доверия до публичного CA). Если сертификату нужно
доверять из конкретного клиента/скрипта — экспортируйте его CA и добавьте в
локальный trust store:

```bash
kubectl get secret cnyrub-predictor-tls -o jsonpath='{.data.ca\.crt}' | base64 -d > ca.crt
```

Если позже появится реальный домен и захочется настоящий сертификат от
Let's Encrypt — меняется только `ClusterIssuer` (`selfSigned` → `acme` с
HTTP-01/DNS-01 challenge через Traefik), сам `IngressRoute` и `values.yaml`
приложения трогать не придётся.

## Переобучение

По расписанию (будни, т.к. MOEX не торгует по выходным) или вручную
(`workflow_dispatch`). Каждый успешный цикл: обучение → gate → promote в
MLflow Registry → новый образ → ArgoCD раскатывает — без ручных `kubectl apply`.

## Смена набора тикеров

`TICKERS` (через запятую) в `src/config.py` / Helm `values.yaml` /
ConfigMap. serve.py тикеры из ConfigMap не читает — он автоматически
подхватывает все модели, реально лежащие в `ARTIFACTS_DIR` внутри образа
(та же логика в `monitor.py`).
