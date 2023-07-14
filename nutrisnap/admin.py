from django.contrib import admin
from .models import Person
from .models import Profile

class PersonAdmin(admin.ModelAdmin):
    search_fields = ['subject']


admin.site.register(Person, PersonAdmin)
# Register your models here.
admin.site.register(Profile)
