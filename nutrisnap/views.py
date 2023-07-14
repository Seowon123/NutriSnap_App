from django.shortcuts import render
from .models import Person


from django.shortcuts import render, redirect
from .image_classification import ImageClassifier
from django.http import HttpResponse
from .models import Profile
import os

model_path = 'trained_models/saved_model.pth'
data_dir = './custom_dataset'
classifier = ImageClassifier(model_path, data_dir)

def index(request):
    return render(request, 'nutrisnap/index.html')

def nutrient_check(request):
    return render(request, 'nutrisnap/nutrient_check.html')

def person_list(request):
    #return render(request, 'nutrisnap/person_list.html')
    person_list = Person.objects.order_by('-create_date')
    context = {'person_list' : person_list}
    return render(request, 'nutrisnap/person_list.html', context)

def person_detail(request):
    person_detail = Person.objects.order_by('-create_date')
    context = {'person_detail' : person_detail}
    return render(request, 'nutrisnap/person_detail.html', context)

def menu_list(request):
    return render(request, 'nutrisnap/menu_list.html')

from .image_classification import ImageClassifier

classifier = ImageClassifier(model_path, data_dir)

# nutrisnap/views.py

from .models import Profile
from .image_classification import ImageClassifier

def classify_image(request):
    if request.method == 'POST':
        if 'image' in request.FILES:
            # 이미지 파일 받기
            image_file = request.FILES['image']

            # nutrisnap_profile 테이블에서 가장 위에 있는 데이터 가져오기
            profile = Profile.objects.order_by('id').first()

            # 이미지 저장하기
            file_path = 'uploaded_image.jpg'
            with open(file_path, 'wb') as f:
                f.write(profile.image.read())

            # 이미지 분류
            classifier = ImageClassifier('trained_models/saved_model.pth', './custom_dataset')
            result = classifier.classify_image(file_path)

            # 분류 결과를 nutrisnap_profile 테이블의 가장 위에 있는 데이터의 id 필드에 저장
            profile.result = result
            profile.save()

            # 이미지 삭제
            os.remove(file_path)

            return render(request, 'nutrisnap/result.html', {'result': result})
        else:
            return HttpResponse('Image not found')

    return render(request, 'nutrisnap/result.html')
def upload(request):
    return render(request,'nutrient_check.html')

def upload_create(request):
    form=Profile()
    try:
        form.image=request.FILES['image']
    except: #이미지가 없어도 그냥 지나가도록-!
        pass
    form.save()
    return redirect('nutrisnap:profile')

def profile(request):
    profile=Profile.objects.all()
    return render(request, 'nutrisnap/profile.html', {'profile': profile})

def result(request):
    profiles = Profile.objects.all().order_by('-id')[:1]
    return render(request, 'nutrisnap/result.html', {'profiles': profiles})
