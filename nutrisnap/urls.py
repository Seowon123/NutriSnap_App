from django.urls import path

from . import views

app_name = 'nutrisnap'

urlpatterns = [
    path('', views.index, name='index'),
    path('check/', views.nutrient_check, name='nutrient_check'),
]