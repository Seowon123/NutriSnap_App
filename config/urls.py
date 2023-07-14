"""config URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from nutrisnap import views
from django.conf.urls.static import static
from django.conf import settings

#from nutrisnap import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('nutrisnap/', include('nutrisnap.urls')),
    path('common/', include('common.urls')),
    path('nutrisnap/', include('nutrisnap.urls')),
    path('upload/', views.upload, name="upload"),
    path('upload_create/', views.upload_create, name="upload_create"),
    path('classify_image/', views.classify_image, name='classify_image'),
]

# 정적 파일 서빙을 위한 URL 패턴 추가
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)