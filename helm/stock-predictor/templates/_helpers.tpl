{{- define "stock-predictor.name" -}}
{{- .Chart.Name -}}
{{- end -}}

{{- define "stock-predictor.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "stock-predictor.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "stock-predictor.labels" -}}
app.kubernetes.io/name: {{ include "stock-predictor.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "stock-predictor.selectorLabels" -}}
app.kubernetes.io/name: {{ include "stock-predictor.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
