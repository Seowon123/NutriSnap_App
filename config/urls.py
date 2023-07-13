from django.contrib import admin
from django.urls import include, path
from nutrisnap import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('nutrisnap.urls')),
    path('nutrisnap/', include('nutrisnap.urls')),
    path('upload/', views.upload, name="upload"),
    path('upload_create/', views.upload_create, name="upload_create"),
    path('classify_image/', views.classify_image, name='classify_image'),
]

# 정적 파일 서빙을 위한 URL 패턴 추가
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
