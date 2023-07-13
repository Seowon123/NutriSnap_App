from . import views
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

app_name = 'nutrisnap'

urlpatterns = [
    path('', views.index, name='index'),
    path('check/', views.nutrient_check, name='nutrient_check'),
    path('classify_image/', views.classify_image, name='classify_image'),
    path('profile/', views.profile, name='profile'),
    path('result/', views.result, name='result'),
]

if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL,
        document_root = settings.MEDIA_ROOT
    )