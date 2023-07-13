from django.contrib import admin
from .models import Nutrient
from .models import Profile

class NutrientAdmin(admin.ModelAdmin):
    search_fields = ['subject']


admin.site.register(Nutrient, NutrientAdmin)
admin.site.register(Profile)

# Register your models here.
