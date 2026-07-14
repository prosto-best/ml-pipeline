{{- define "cnyrub-predictor.name" -}}
{{- .Chart.Name -}}
{{- end -}}

{{- define "cnyrub-predictor.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "cnyrub-predictor.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "cnyrub-predictor.labels" -}}
app.kubernetes.io/name: {{ include "cnyrub-predictor.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "cnyrub-predictor.selectorLabels" -}}
app.kubernetes.io/name: {{ include "cnyrub-predictor.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
