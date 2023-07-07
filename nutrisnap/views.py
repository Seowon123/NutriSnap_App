from django.shortcuts import render
from django.shortcuts import render
from .models import Nutrient


def index(request):
    nutrient_list = Nutrient.objects.order_by('-create_date')
    context = {'nutrient_list': nutrient_list}
    return render(request, 'nutrisnap/nutrient_list.html', context)

def detail(request, nutrient_id):
    nutrient = Nutrient.objects.get(id=nutrient_id)
    context = {'nutrient': nutrient}
    return render(request, 'nutrisnap/nutrient_detail.html', context)