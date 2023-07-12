from django.shortcuts import render



def index(request):
    return render(request, 'nutrisnap/index.html')

def nutrient_check(request):
    return render(request, 'nutrisnap/nutrient_check.html')
