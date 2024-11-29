from django.urls import path
from .views import RandomForestPredictionAPIView, GradientBoostingPredictionAPIView, api_home

urlpatterns = [
    path('', api_home, name='api_home'),
    path('predict/random_forest/', RandomForestPredictionAPIView.as_view(), name='random_forest_predict'),
    path('predict/gradient_boosting/', GradientBoostingPredictionAPIView.as_view(), name='gradient_boosting_predict'),
]
