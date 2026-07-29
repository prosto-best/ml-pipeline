{{/*
Общее имя релиза + чарта, используем в имени ресурсов
*/}}
{{- define "simple-app.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Стандартные лейблы, которые вешаем на все ресурсы
*/}}
{{- define "simple-app.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Селектор — используется и в Deployment.spec.selector, и в Service.spec.selector
*/}}
{{- define "simple-app.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
