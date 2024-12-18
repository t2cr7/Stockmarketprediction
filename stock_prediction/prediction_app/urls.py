from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Home page
    path('about/', views.about, name='about'),  # About Us page
    path('tickers_info/', views.tickers_info, name='tickers_info'),  # Tickers Info page
    path('insertintotable/', views.insertintotable, name='insertintotable'),  # Prediction result page
]
