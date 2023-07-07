from django.urls import path

from . import views

urlpatterns = [
    path('', views.index),
    path('<int:nutrisnap_id>/', views.detail),
]