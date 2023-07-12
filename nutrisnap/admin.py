from django.contrib import admin
from .models import Nutrient

class NutrientAdmin(admin.ModelAdmin):
    search_fields = ['subject']


admin.site.register(Nutrient, NutrientAdmin)
# Register your models here.
